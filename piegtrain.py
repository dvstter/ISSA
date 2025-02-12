# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import wandb

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, total_physics, cfg):
  # fill missing hydra's parameters for agent and encoder, and then instantiate them
  cfg.obs_shape = obs_spec.shape
  cfg.action_shape = action_spec.shape
  cfg.encoder.obs_shape = obs_spec.shape
  cfg.total_physics = total_physics
  return hydra.utils.instantiate(cfg)


class Workspace:
  def __init__(self, cfg):
    self.work_dir = Path.cwd()
    print(f'workspace: {self.work_dir}')

    self.cfg = cfg
    utils.set_seed_everywhere(cfg.seed)
    self.device = torch.device(cfg.device)
    self.setup()

    self.agent = make_agent(self.train_env.observation_spec(),
                            self.train_env.action_spec(),
                            self.train_env.num_physics_state(),
                            self.cfg.agent)
    self.timer = utils.Timer()
    self._global_step = 0
    self._global_episode = 0

  def setup(self):
    # create logger
    encoder_name = self.cfg.encoder._target_.strip().split('.')[1].lower()[:-7]
    physics_aux = self.cfg.agent.physics_aux_sequence
    if physics_aux not in ['none', 'without_vision', 'all']:
      physics_aux = ','.join([str(x) for x in physics_aux])
    physics_aux = 'phy_' + physics_aux
    if self.cfg.use_wandb:
      exp_name = [encoder_name,
                  str(self.cfg.seed),
                  'bat' + str(self.cfg.replay_buffer.batch_size),
                  'fea' + str(self.cfg.agent.feature_dim),
                  'hid' + str(self.cfg.agent.hidden_dim)]
      if encoder_name == 'simclr':
        exp_name += ['lay' + str(self.cfg.encoder.applied_layers)]
      exp_name += [physics_aux]
      exp_name = '_'.join(exp_name)
      group_name = '_'.join([str(self.cfg.wandb_group_prefix), str(self.cfg.task_name)])
      wandb.init(project="urlb_finetune", group=group_name, name=exp_name)
    self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
    # create envs
    self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
    self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
    # create replay buffer
    data_specs = (self.train_env.observation_spec(),
                  self.train_env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'),
                  specs.Array((self.train_env.num_physics_state(),), np.float32, 'physics'))

    self.replay_buffer = hydra.utils.instantiate(self.cfg.replay_buffer, data_specs=data_specs)

    self.video_recorder = VideoRecorder( self.work_dir if self.cfg.save_video else None)
    self.train_video_recorder = TrainVideoRecorder( self.work_dir if self.cfg.save_train_video else None)

  @property
  def global_step(self):
    return self._global_step

  @property
  def global_episode(self):
    return self._global_episode

  @property
  def global_frame(self):
    return self.global_step * self.cfg.action_repeat

  def eval(self):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

    while eval_until_episode(episode):
      time_step = self.eval_env.reset()
      self.video_recorder.init(self.eval_env, enabled=(episode == 0))
      while not time_step.last():
        with torch.no_grad(), utils.eval_mode(self.agent):
          action = self.agent.act(time_step.observation, time_step.physics, self.global_step, eval_mode=True)
        time_step = self.eval_env.step(action)
        self.video_recorder.record(self.eval_env)
        total_reward += time_step.reward
        step += 1

      episode += 1
      self.video_recorder.save(f'{self.global_frame}.mp4')

    with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
      log('episode_reward', total_reward / episode)
      log('episode_length', step * self.cfg.action_repeat / episode)
      log('episode', self.global_episode)
      log('step', self.global_step)

  def train(self):
    # predicates
    # print('self.cfg.num_train_frames:', self.cfg.num_train_frames)
    train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
    seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
    eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

    episode_step, episode_reward = 0, 0
    time_step = self.train_env.reset()
    self.replay_buffer.add(time_step)
    self.train_video_recorder.init(time_step.observation)
    metrics = None
    while train_until_step(self.global_step):
      # update metrics and reset environment when this time step is last
      if time_step.last():
        self._global_episode += 1
        self.train_video_recorder.save(f'{self.global_frame}.mp4')
        # wait until all the metrics schema is populated
        if metrics is not None:
          # log stats
          elapsed_time, total_time = self.timer.reset()
          episode_frame = episode_step * self.cfg.action_repeat
          with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
            log('fps', episode_frame / elapsed_time)
            log('total_time', total_time)
            log('episode_reward', episode_reward)
            log('episode_length', episode_frame)
            log('episode', self.global_episode)
            log('buffer_size', len(self.replay_buffer))
            log('step', self.global_step)

        # reset env
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        # try to save snapshot
        if self.cfg.save_snapshot and (self.global_step % int(1e5) == 0):
          self.save_snapshot()
        episode_step = 0
        episode_reward = 0

      # try to evaluate
      if eval_every_step(self.global_step):
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.eval()

      # sample action
      with torch.no_grad(), utils.eval_mode(self.agent):
        action = self.agent.act(time_step.observation, time_step.physics, self.global_step, eval_mode=False)

      # wont update agent before num_seed_frames reached
      if not seed_until_step(self.global_step):
        metrics = self.agent.update(self.replay_buffer, self.global_step)
        self.logger.log_metrics(metrics, self.global_frame, ty='train')

      # take env step
      time_step = self.train_env.step(action)
      episode_reward += time_step.reward
      self.replay_buffer.add(time_step)
      self.train_video_recorder.record(time_step.observation)
      episode_step += 1
      self._global_step += 1

  def save_snapshot(self):
    snapshot = self.work_dir / 'snapshot.pt'
    keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
    payload = {k: self.__dict__[k] for k in keys_to_save}
    with snapshot.open('wb') as f:
      torch.save(payload, f)

  def load_snapshot(self):
    snapshot = self.work_dir / 'snapshot.pt'
    with snapshot.open('rb') as f:
      payload = torch.load(f)
    for k, v in payload.items():
      self.__dict__[k] = v


@hydra.main(version_base=None, config_path='cfgs', config_name='piegconfig')
def main(cfg):
  from piegtrain import Workspace as W
  root_dir = Path.cwd()
  workspace = W(cfg)
  snapshot = root_dir / 'snapshot.pt'
  if snapshot.exists() and cfg.load_snapshot:
    print(f'resuming: {snapshot}')
    workspace.load_snapshot()
  workspace.train()


if __name__ == '__main__':
  main()
