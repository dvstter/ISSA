# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
import torch.distributions as pyd

import utils
from pieg import PIEGAgent

class ImitationAgent(PIEGAgent):
    def __init__(self, obs_shape, action_shape, device, encoder, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 total_physics, physics_aux_sequence):
        if physics_aux_sequence not in ['all', 'without_vision']:
            raise Exception('physics_aux_sequence should only be all or without_vision')
        self.teacher = PIEGAgent(obs_shape, action_shape, device, encoder, lr, feature_dim, hidden_dim,
                                 critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule, stddev_clip,
                                 use_tb, total_physics, physics_aux_sequence)
        super(ImitationAgent, self).__init__(obs_shape, action_shape, device, encoder, lr, feature_dim, hidden_dim,
                                             critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule,
                                             stddev_clip, use_tb, total_physics, 'none')
        self._pure_vision = False
        self.total_physics = total_physics

    def train(self, training=True):
        super(ImitationAgent, self).train(training)
        self.teacher.train(training)

    @property
    def pure_vision(self):
        return self._pure_vision

    @pure_vision.setter
    def pure_vision(self, value):
        self._pure_vision = value

    def act(self, obs, physics, step, eval_mode):
        if self.pure_vision:
            return super(ImitationAgent, self).act(obs, physics, step, eval_mode)
        else:
            return self.teacher.act(obs, physics, step, eval_mode)

    def update_critic(self, obs, physics, action, reward, discount, next_obs, next_physics, step, aug_obs):
        if self.pure_vision:
            return super(ImitationAgent, self).update_critic(obs, physics, action, reward, discount, next_obs, next_physics, step, aug_obs)

        # firstly update teacher's critic
        metrics = dict()

        with torch.no_grad():
            t_next_obs = self.teacher.append_physics(next_obs, next_physics)
            stddev = utils.schedule(self.teacher.stddev_schedule, step)
            dist = self.teacher.actor(t_next_obs, stddev)
            next_action = dist.sample(clip=self.teacher.stddev_clip)
            target_Q1, target_Q2 = self.teacher.critic_target(t_next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        t_obs = self.teacher.append_physics(obs, physics)
        t_aug_obs = self.teacher.append_physics(aug_obs, physics)
        Q1, Q2 = self.teacher.critic(t_obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.teacher.critic(t_aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize teacher's encoder and critic
        self.teacher.encoder_opt.zero_grad(set_to_none=True)
        self.teacher.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.teacher.critic_opt.step()
        self.teacher.encoder_opt.step()

        # then, make agent learn something from the teacher
        target_Q = target_Q.detach().clone()
        Q1, Q2 = self.critic(obs.detach().clone(), action)
        aug_Q1, aug_Q2 = self.critic(aug_obs.detach().clone(), action)
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q) +
                      F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q))

        if self.use_tb:
            metrics['sl_q1'] = Q1.mean().item()
            metrics['sl_q2'] = Q2.mean().item()
            metrics['sl_critic_loss'] = critic_loss.mean().item()

        # optimize agent's encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, physics, step):
        if self.pure_vision:
            return super(ImitationAgent, self).update_actor(obs, physics, step)

        metrics = dict()

        # firstly optimize teacher's actor
        t_obs = self.teacher.append_physics(obs, physics)
        stddev = utils.schedule(self.teacher.stddev_schedule, step)
        dist = self.teacher.actor(t_obs, stddev)
        action = dist.sample(clip=self.teacher.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.teacher.critic(t_obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.teacher.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.teacher.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()


        # then optimize agent's actor
        target_loc = dist.loc.detach().clone()
        dist = self.actor(obs.detach().clone(), stddev)
        # kl distance loss function
#        target_dist = utils.TruncatedNormal(dist.loc.detach().clone(), dist.scale.detach().clone())
#        actor_loss = pyd.kl_divergence(dist, target_dist).mean()

        # only consider the location of the normal distribution
        actor_loss = F.mse_loss(target_loc, dist.loc)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['sl_actor_loss'] = actor_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = super(ImitationAgent, self).update(replay_iter, step)
        if not self.pure_vision:
            utils.soft_update_params(self.teacher.critic, self.teacher.critic_target, self.teacher.critic_target_tau)
        return metrics
