# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import random_overlay
from pieg import RandomShiftsAug
from pieg import Actor as piegActor
from pieg import Critic as piegCritic

class Actor(nn.Module):
    def __init__(self, vision_repr_dim, inner_state_repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.pure_vision = False
        self.v_actor = piegActor(vision_repr_dim, action_shape, feature_dim, hidden_dim)
        self.i_actor = piegActor(inner_state_repr_dim, action_shape, feature_dim, hidden_dim)

    def change_mode_pure_vision(self):
        self.pure_vision = True
        self.i_actor.train(False)
        self.i_actor.requires_grad_(False)

    def forward(self, obs, phy, std):
        forward_func = lambda actor, x: actor.policy(actor.trunk(x))
        temp = forward_func(self.v_actor, obs)
        if not self.pure_vision:
            temp += forward_func(self.i_actor, phy)

        std = torch.ones_like(temp) * std
        temp = torch.tanh(temp)
        dist = utils.TruncatedNormal(temp, std)
        return dist

class Critic(nn.Module):
    def __init__(self, vision_repr_dim, inner_state_repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.pure_vision = False
        self.v_critic = piegCritic(vision_repr_dim, action_shape, feature_dim, hidden_dim)
        self.i_critic = piegCritic(inner_state_repr_dim, action_shape, feature_dim, hidden_dim)

    def change_mode_pure_vision(self):
        self.pure_vision = True
        self.i_critic.train(False)
        self.i_critic.requires_grad_(False)

    def forward(self, obs, phy, action):
#        q1, q2 = self.v_critic(obs, action)
#        if not self.pure_vision:
#            q1_p, q2_p  = self.i_critic(phy, action)
#            return q1+q1_p, q2+q2_p
#        else:
#            return q1, q2

        # another way to implement
        h = self.v_critic.trunk(obs)
        if not self.pure_vision:
#            h += self.i_critic.trunk(phy) # fatal error, += is inplace operation, will induce loss.backward operation failed
            h = h + self.i_critic.trunk(phy)

        h_action = torch.cat([h, action], dim=-1)
        q1 = self.v_critic.Q1(h_action)
        q2 = self.v_critic.Q2(h_action)
        return q1, q2

class ContinueTrainAgent:
    def __init__(self, obs_shape, action_shape, device, encoder, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 total_physics, physics_aux_sequence):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.pure_vision = False
        try:
            if str(physics_aux_sequence).lower() == 'all':
                self.physics_aux_sequence = list(range(total_physics))
            else:
                self.physics_aux_sequence = list(physics_aux_sequence)
        except Exception as e:
            print(f'instantiated ContinueTrainAgent with parameter physics_aux_sequence={physics_aux_sequence} is forbidden.'
                  f'this parameter should be \'all\' or a list.')
            exit(1)

        self.encoder = encoder.to(device)
        vision_repr_dim = self.encoder.repr_dim
        inner_state_repr_dim = len(self.physics_aux_sequence)

        self.actor = Actor(vision_repr_dim, inner_state_repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(vision_repr_dim, inner_state_repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(vision_repr_dim, inner_state_repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        print(f'instantiated continue_train agent with vision_repr_dim {vision_repr_dim} inner_state_repr_dim {inner_state_repr_dim} physics_aux_sequence {self.physics_aux_sequence}')

    def change_mode_pure_vision(self):
        self.pure_vision = True
        self.actor.change_mode_pure_vision()
        self.critic.change_mode_pure_vision()
        self.critic_target.change_mode_pure_vision()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def prepare_physics(self, physics, dtype):
        physics = torch.as_tensor(physics, dtype=dtype)
        physics = physics.unsqueeze(0) if physics.ndim == 1 else physics
        physics = physics[:, self.physics_aux_sequence].to(self.device)
        return physics

    def act(self, obs, physics, step, eval_mode):
        obs = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        physics = self.prepare_physics(physics, obs.dtype)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, physics, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, physics, action, reward, discount, next_obs, next_physics, step, aug_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, next_physics, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_physics, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, physics, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, physics, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, physics, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, physics, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, physics, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, physics, next_physics = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)

        # strong augmentation
        aug_obs = self.encoder(random_overlay(original_obs))

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # prepare physics
        physics = self.prepare_physics(physics, obs.dtype)
        next_physics = self.prepare_physics(next_physics, next_obs.dtype)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, physics, action, reward, discount, next_obs, next_physics, step, aug_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), physics.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
