import torch
import torch.distributions as D

import world_models.common as common


class Agent(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, act_range, config):
        super(Agent, self).__init__()
        # config
        self.config = config
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim

        # models
        self.world_model = common.WorldModel(obs_dim, act_dim, config)
        self.ensemble = common.Ensemble(act_dim, config)

        self.expl_actor = common.Actor(act_dim, act_range, config)
        self.expl_critic = common.TwoHotSymlogMLP(config)
        self.expl_slow_critic = common.TwoHotSymlogMLP(config).requires_grad_(False)

        self.act_dim = act_dim
        self.act_range = act_range
        self.task_actor = common.Actor(act_dim, act_range, config)
        self.task_critic = common.TwoHotSymlogMLP(config)
        self.task_slow_critic = common.TwoHotSymlogMLP(config).requires_grad_(False)

        # training
        wm_opt = {'eps': config.model_eps, 'weight_decay': config.weight_decay}
        ac_opt = {'eps': config.ac_eps, 'weight_decay': config.weight_decay}

        self.world_model_optim = torch.optim.Adam(self.world_model.parameters(), config.model_lr, **wm_opt)
        self.ensemble_optim = torch.optim.Adam(self.ensemble.parameters(), config.model_lr, **wm_opt)

        self.expl_actor_optim = torch.optim.Adam(self.expl_actor.parameters(), config.actor_lr, **ac_opt)
        self.expl_critic_optim = torch.optim.Adam(self.expl_critic.parameters(), config.critic_lr, **ac_opt)

        self.task_actor_optim = torch.optim.Adam(self.task_actor.parameters(), config.actor_lr, **ac_opt)
        self.task_critic_optim = torch.optim.Adam(self.task_critic.parameters(), config.critic_lr, **ac_opt)

        self._updates = 0
        self.actor, self.actor_optim = None, None
        self.critic, self.slow_critic, self.critic_optim = None, None, None

        # ditto
        self.expert_states, self.h_last = None, None

        # utility
        self.reward_ema = common.RewardEMA(config.device)
        self.device = config.device
        self.set_actor_critic()

    # --------------------------------------------------------------------------------------------------------------
    # Environment interaction

    def __call__(self, h_t, obs):
        with torch.no_grad():
            obs = common.symlog(torch.Tensor(obs).to(self.device))
            z_t = self.world_model.encode(h_t, obs)
            action = self.actor(torch.cat((h_t, z_t), dim=-1)).sample()
            h_t1 = self.world_model.forward(h_t, z_t, action)
        return h_t1, action

    def predict(self, h_t, obs):
        with torch.no_grad():
            h_t1, action = self(h_t, obs)
            z_t1 = self.world_model.dynamics(h_t1)
            obs_1, reward_1, cont_1 = self.world_model.predict(torch.cat((h_t1, z_t1), dim=-1))
        return (obs_1, reward_1, cont_1), h_t1, action

    # --------------------------------------------------------------------------------------------------------------
    # Training

    def train_step(self, step, replay, should_train):
        if self.config.Plan2Explore and step >= self.config.explore_steps:
            print('Ending Plan2Explore')
            self.config.Plan2Explore = False
        self.set_actor_critic()

        if should_train:
            data, states, rssm_info = self.train_world_model(replay)
            ensemble_info = self._train_ensemble(data, states) if self.config.Plan2Explore else {'ensemble_loss': 0}

            states = torch.flatten(states['state'], 0, 1).detach()
            ac_info = self._train_actor_critic(states)
            return {**rssm_info, **ensemble_info, **ac_info}
        else:
            return {}

    def ditto_step(self, replay):
        self.expert_states = replay.sample(self.config.ditto_batch_size, self.config.imag_horizon + 1)['state'].detach()
        return self._train_actor_critic(self.expert_states[0])

    def encode_expert_data(self, replay):
        data = replay.get_all()
        h_init = self._init_deter(1).squeeze(0)
        return self._encode_data(data, h_init, {'state': []})[0]

    # --------------------------------------------------------------------------------------------------------------
    # World Model

    def train_world_model(self, replay):
        # (seq_length, batch_size, obs_dim)
        data = next(replay)
        h_init = self._init_deter(1).squeeze(0) if replay.idx == 1 else self.h_last

        states, h_last = self._encode_data(data, h_init, {'state': [], 'post': [], 'prior': []})
        self.h_last = h_last.detach()

        # kl divergences
        d = common.CategoricalDist
        kl = lambda x, y: torch.clip(D.kl.kl_divergence(x.dist, y.dist), min=1.0)

        # losses
        pred_loss = -self.world_model.log_probs(data, states['state']).mean()
        dyn_loss = self.config.beta_dyn * kl(d(states['post'].detach()), d(states['prior'])).mean()
        repr_loss = self.config.beta_repr * kl(d(states['post']), d(states['prior'].detach())).mean()
        loss = pred_loss + dyn_loss + repr_loss

        # update weights
        self.world_model_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.model_grad_clip)
        self.world_model_optim.step()

        info = {'pred_loss': pred_loss.item(), 'kl_loss': (dyn_loss + repr_loss).item()}
        return data, states, info

    def _encode_data(self, data, h_init, states):
        batch_length = data['obs'].shape[0]
        h_t = h_init

        for t in range(batch_length):
            obs = data['obs'][t]
            post, prior = self.world_model.get_z_dists(h_t, obs)
            z_t = post.sample()
            latents = {'state': torch.cat((h_t, z_t), dim=-1), 'post': post.logits, 'prior': prior.logits}
            for k, v in latents.items():
                if k in states:
                    states[k].append(v)
            if t < batch_length - 1:
                h_t = self.world_model.forward(h_t, z_t, data['action'][t])

        states = {k: torch.stack(v, dim=0) for k, v in states.items()}
        return states, h_t

    # --------------------------------------------------------------------------------------------------------------
    # Ensemble

    def _train_ensemble(self, data, states):
        self.world_model.requires_grad_(False)

        actions = data['action'][:, :self.config.ensemble_size]
        states = states['state'][:, :self.config.ensemble_size].detach()
        pred_states = self.ensemble(torch.cat((states[:-1], actions[:-1]), dim=-1))
        loss = (pred_states - states[1:]).square().mean()

        # update weights
        self.ensemble_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), self.config.model_grad_clip)
        self.ensemble_optim.step()

        self.world_model.requires_grad_(True)

        return {'ensemble_loss': loss}

    # --------------------------------------------------------------------------------------------------------------
    # Actor Critic

    def _train_actor_critic(self, states_0):
        self.world_model.requires_grad_(False)
        self.ensemble.requires_grad_(False)

        states_t = states_0
        act_t = self.actor(states_t).sample()

        # imagine rollouts
        states, actions, action_log_probs = [], [], []
        info = {'entropy': 0, 'act_size': 0, 'act_std': 0}
        for _ in range(self.config.imag_horizon):
            states_t = self.world_model.step(states_t, act_t)
            actor = self.actor(states_t)
            act_t = actor.sample()
            log_prob = actor.log_prob(act_t.detach())

            states.append(states_t)
            actions.append(act_t)
            action_log_probs.append(log_prob)

            info['entropy'] += actor.entropy().mean() / self.config.imag_horizon
            info['act_size'] += act_t.square().mean() / self.config.imag_horizon
            info['act_std'] += actor.stddev.mean() / self.config.imag_horizon

        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        action_log_probs = torch.stack(action_log_probs, dim=0).unsqueeze(-1)

        # calculate rewards
        if self.config.Plan2Explore:
            rewards = self.ensemble.get_variance(torch.cat((states, actions), dim=-1))
        elif self.config.ditto:
            rewards = self._calculate_ditto_rewards(states)
        else:
            rewards = self.world_model.reward(states)
        # gammas = self.config.gamma * self.world_model.cont(states)
        gammas = self.config.gamma * torch.ones_like(rewards)
        # weights = torch.cumprod(gammas, dim=0).detach()  # to account for episode termination
        weights = torch.ones_like(gammas).detach()

        # calculate value targets
        policy_loss, entropy_loss, value_targets = self._calculate_policy_loss(
            states, action_log_probs, rewards, gammas, weights, info['entropy'])
        value_loss, values = self._calculate_value_loss(states, value_targets, weights)
        loss = policy_loss + value_loss - entropy_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.ac_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.ac_grad_clip)
        self.actor_optim.step()
        self.critic_optim.step()

        self._update_slow_critic()
        self.world_model.requires_grad_(True)
        self.ensemble.requires_grad_(True)

        return {'policy_loss': policy_loss,
                'value_loss': value_loss,
                'imag_reward': rewards.mean(),
                'imag_values': values.mean(),
                'imag_value_targets': value_targets.mean(),
                'reward_ema': {'05': self.reward_ema.values[0], '95': self.reward_ema.values[1]},
                **info}

    def _calculate_policy_loss(self, states, action_log_probs, rewards, gammas, weights, entropy):
        self.critic.requires_grad_(False)

        value_targets = []
        values = self.critic(states).mode()
        for t in reversed(range(self.config.imag_horizon - 1)):
            if t == self.config.imag_horizon - 2:
                target = values[t + 1]
            else:
                target = (1 - self.config.lam) * values[t + 1] + self.config.lam * value_targets[-1]
            value_targets.append(rewards[t] + gammas[t] * target)

        value_targets = torch.stack(value_targets[::-1], dim=0)
        offset, scale = self.reward_ema(value_targets)
        norm_targets = (value_targets - offset) / scale
        norm_base = (values[:-1] - offset) / scale
        adv = norm_targets - norm_base

        if self.config.policy_gradient == 'dynamics':
            actor_target = adv
        elif self.config.policy_gradient == 'reinforce':
            actor_target = action_log_probs[:-1] * adv.detach()
        else:
            raise NotImplementedError(f'Unknown policy gradient: {self.config.policy_gradient}')

        policy_loss = -(weights[:-1] * actor_target).mean()
        entropy_loss = self.config.entropy_coeff * entropy

        self.critic.requires_grad_(True)
        return policy_loss, entropy_loss, value_targets

    def _calculate_value_loss(self, states, value_targets, weights):
        self.actor.requires_grad_(False)

        values = self.critic(states[:-1].detach())
        slow_critic = self.slow_critic(states[:-1].detach())
        value_loss = -values.log_prob(value_targets.detach())
        value_loss -= values.log_prob(slow_critic.mode().detach())
        value_loss = (weights[:-1] * value_loss.unsqueeze(-1)).mean()

        self.actor.requires_grad_(True)
        return value_loss, values.mode()

    def _update_slow_critic(self):
        if self._updates % self.config.slow_critic_update == 0:
            mix = self.config.slow_critic_fraction
            for s, d in zip(self.critic.parameters(), self.slow_critic.parameters()):
                d.data = mix * s.data + (1 - mix) * d.data
        self._updates += 1

    def _calculate_ditto_rewards(self, states):
        # dot product
        reward = torch.sum(self.expert_states[1:, :, :self.config.h_dim] * states[..., :self.config.h_dim], dim=-1)
        reward /= (torch.maximum(torch.norm(self.expert_states[1:, :, :self.config.h_dim], dim=-1),
                                 torch.norm(states[..., :self.config.h_dim], dim=-1)) ** 2)
        return reward.unsqueeze(-1)

    # --------------------------------------------------------------------------------------------------------------
    # Utility

    def set_actor_critic(self):
        if self.config.Plan2Explore:
            self.actor = self.expl_actor
            self.critic = self.expl_critic
            self.slow_critic = self.expl_slow_critic
            self.actor_optim = self.expl_actor_optim
            self.critic_optim = self.expl_critic_optim
        else:
            self.actor = self.task_actor
            self.critic = self.task_critic
            self.slow_critic = self.task_slow_critic
            self.actor_optim = self.task_actor_optim
            self.critic_optim = self.task_critic_optim

    def _init_deter(self, size):
        if self.config.init_deter == 'zero':
            return torch.zeros((size, self.h_dim)).to(self.device)
        elif self.config.init_deter == 'normal':
            return 0.01 * torch.randn((size, self.h_dim)).to(self.device)
        else:
            raise NotImplementedError(f'Unknown init_deter: {self.config.init_deter}')
