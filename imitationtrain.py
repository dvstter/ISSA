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
import torch
import utils

from piegtrain import Workspace as piegWorkspace

class Workspace(piegWorkspace):
  def train(self):
    train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
    seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
    eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
    train_util_pure_vision = utils.Until(self.cfg.num_train_frames * self.cfg.pure_vision_ratio, self.cfg.action_repeat)

    episode_step, episode_reward = 0, 0
    time_step = self.train_env.reset()
    self.replay_buffer.add(time_step)
    self.train_video_recorder.init(time_step.observation)
    metrics = None
    mode_pure_vision = False
    while train_until_step(self.global_step):
      # try to change mode to pure vision
      if not train_util_pure_vision(self.global_step) and not mode_pure_vision:
        mode_pure_vision = True
        print(f'current step {self.global_step} change mode to pure vision.')
        self.agent.pure_vision = True

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

@hydra.main(version_base=None, config_path='cfgs', config_name='imitationconfig')
def main(cfg):
  from imitationtrain import Workspace as W
  root_dir = Path.cwd()
  workspace = W(cfg)
  snapshot = root_dir / 'snapshot.pt'
  if snapshot.exists() and cfg.load_snapshot:
    print(f'resuming: {snapshot}')
    workspace.load_snapshot()
  workspace.train()


if __name__ == '__main__':
  main()
