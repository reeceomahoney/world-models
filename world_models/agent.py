import math
from collections import defaultdict

import torch
import torch.distributions as D

import world_models.common as common


class Agent(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, act_range, logger, config):
        super(Agent, self).__init__()
        # config
        self.config = config
        self.obs_dim = obs_dim
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim

        # models
        self.world_model = common.WorldModel(obs_dim, act_dim, config)
        self.ensemble = common.Ensemble(act_dim, config)

        self.act_dim = act_dim
        self.act_range = act_range
        self.task_actor = common.Actor(act_dim, act_range, config)
        self.expl_actor = common.Actor(act_dim, act_range, config)

        if config.critic_model == 'Gaussian':
            self.task_critic = common.GaussianMLP(config)
            self.task_slow_critic = common.GaussianMLP(
                config).requires_grad_(False)
            self.expl_critic = common.GaussianMLP(config)
            self.expl_slow_critic = common.GaussianMLP(
                config).requires_grad_(False)
        elif config.critic_model == 'TwoHot':
            self.task_critic = common.TwoHotSymlogMLP(config)
            self.task_slow_critic = common.TwoHotSymlogMLP(
                config).requires_grad_(False)
            self.expl_critic = common.TwoHotSymlogMLP(config)
            self.expl_slow_critic = common.TwoHotSymlogMLP(
                config).requires_grad_(False)

        # training
        wm_opt = {'eps': config.model_eps, 'weight_decay': config.weight_decay}
        ac_opt = {'eps': config.ac_eps, 'weight_decay': config.weight_decay}

        self.world_model_optim = torch.optim.Adam(
            self.world_model.parameters(), config.model_lr, **wm_opt)
        self.ensemble_optim = torch.optim.Adam(
            self.ensemble.parameters(), config.model_lr, **wm_opt)

        self.expl_actor_optim = torch.optim.Adam(
            self.expl_actor.parameters(), config.actor_lr, **ac_opt)
        self.expl_critic_optim = torch.optim.Adam(
            self.expl_critic.parameters(), config.critic_lr, **ac_opt)

        self.task_actor_optim = torch.optim.Adam(
            self.task_actor.parameters(), config.actor_lr, **ac_opt)
        self.task_critic_optim = torch.optim.Adam(
            self.task_critic.parameters(), config.critic_lr, **ac_opt)

        self._updates = 0
        self.actor, self.actor_optim = None, None
        self.critic, self.slow_critic, self.critic_optim = None, None, None
        self.value_targets = None

        self.states, self.states_logits, self.rewards = None, None, None
        self.action, self.action_log_probs = None, None
        self.gammas, self.weights = None, None

        self.logger = logger

        # ditto
        self.expert_states, self.expert_actions, self.h_last = None, None, None

        # amp
        self.disc = common.Discriminator(config)
        self.disc_optim = torch.optim.Adam(
            self.disc.parameters(), config.disc_lr, **ac_opt)
        self.state_last = None

        # utility
        self.reward_ema = common.RewardEMA(config.device)
        self.device = config.device
        self.set_actor_critic()

    # -------------------------------------------------------------------------
    # Environment interaction

    def __call__(self, h_t, obs, deterministic=False):
        with torch.no_grad():
            obs = common.symlog(torch.Tensor(obs).to(self.device))
            z_t = self.world_model.encode(h_t, obs)
            action = self.actor(torch.cat((h_t, z_t), dim=-1)).sample()
            if deterministic:
                action = self.actor(torch.cat((h_t, z_t), dim=-1)).mode
            h_t1 = self.world_model.forward(h_t, z_t, action)
        return h_t1, action

    def predict(self, h_t, obs):
        with torch.no_grad():
            h_t1, action = self(h_t, obs, deterministic=True)
            z_t1 = self.world_model.dynamics(h_t1)
            state = torch.cat((h_t1, z_t1), dim=-1)
            obs_1, reward_1, cont_1 = self.world_model.predict(state)
        return (obs_1, reward_1, cont_1), state, action

    # -------------------------------------------------------------------------
    # Training

    # def train_step(self, step, replay, should_train):
    #     if self.config.Plan2Explore and step >= self.config.explore_steps:
    #         print('Ending Plan2Explore')
    #         self.config.Plan2Explore = False
    #     self.set_actor_critic()

    #     if should_train:
    #         states = self.train_world_model(replay)
    #         if self.config.Plan2Explore:
    #             ensemble_info = self._train_ensemble(data, states)
    #         else:
    #             ensemble_info = {}

    #         states = torch.flatten(states['state'], 0, 1).detach()
    #         ac_info = self._train_actor_critic(states)
    #         return {**rssm_info, **ensemble_info, **ac_info}
    #     else:
    #         return {}

    def ditto_step(self, expert_sampler):
        """
        Samples and encodes expert data then uses this to train the
        actor-critic using DITTO.

        Args:
            expert_sampler (ExpertSampler): sampler for expert data

        Returns:
            info (dict): training info
        """
        # sample and encode expert data
        expert_sample = expert_sampler.sample(self.config.imag_horizon + 1,
                                              self.config.ditto_batch_size)
        expert_states = self.encode_data(expert_sample)[0]
        expert_states = {k: v.detach() for k, v in expert_states.items()}

        # shift actions to match states (data is stored s_t, a_t+1)
        expert_states['action'][1:] = expert_states['action'][:-1].clone()
        self.expert_actions = expert_states['action']

        assert expert_states['state'].shape == (
            self.config.imag_horizon + 1, self.config.ditto_batch_size,
            self.h_dim + self.z_dim)
        assert expert_states['action'].shape == (
            self.config.imag_horizon + 1, self.config.ditto_batch_size,
            self.act_dim)
        assert expert_states['state'].grad_fn is None
        assert expert_states['action'].grad_fn is None

        # augment states if the ditto reward uses logits
        if self.config.ditto_state == 'logits':
            self.expert_states = torch.cat((
                expert_states['state'][..., :self.config.h_dim],
                expert_states['post']), dim=-1)
        else:
            self.expert_states = expert_states['state']

        # initialise actor critic training at the first expert state
        return self._train_actor_critic(self.expert_states[0])

    # -------------------------------------------------------------------------
    # World Model

    def train_world_model(self, expert_sampler, train=True):
        """
        Trains the world model using sequentially sampled data.

        Args:
            expert_sampler (ExpertSampler): sampler for expert data
            train (bool): whether to train the world model
        """
        # sample data sequentially to preserve temporal correlations
        data = next(expert_sampler)

        assert data['obs'].shape == (
            self.config.batch_length, self.config.ditto_wm_batch_size,
            self.obs_dim), data['obs'].shape

        # initialise non-initial states with the last hidden state
        if expert_sampler.idx == 1:
            h_init = None
        else:
            h_init = self.h_last

        states, h_last = self.encode_data(data, h_init)
        self.h_last = h_last.detach()

        assert states['state'].shape == (
            self.config.batch_length, self.config.ditto_wm_batch_size,
            self.h_dim + self.z_dim)

        # compute losses
        d = self._get_z_dist()
        pred_loss = -self.world_model.log_probs(data, states['state']).mean()
        dyn_loss = self.config.beta_dyn * self.kl_div(
            d(states['post'].detach()), d(states['prior'])).mean()
        repr_loss = self.config.beta_repr * self.kl_div(
            d(states['post']), d(states['prior'].detach())).mean()
        loss = pred_loss + dyn_loss + repr_loss

        # update weights
        if train:
            self.world_model_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(),
                                           self.config.model_grad_clip)
            self.world_model_optim.step()

        # log
        self.logger.log('world_model',
                        {'pred_loss': pred_loss.item(),
                         'kl_loss': (dyn_loss + repr_loss).item()})

        if self.config.z_dist == 'Categorical':
            self.logger.log(
                'world_model',
                {'post_entropy': d(states['post']).entropy().mean(),
                 'prior_entropy': d(states['prior']).entropy().mean()})
        elif self.config.z_dist == 'Gaussian':
            z_std = states['prior'][..., self.z_dim:]
            self.logger.log('world_model',
                            {'z_std': z_std.mean().item(),
                             'z_std_max': z_std.max().item(),
                             'z_std_min': z_std.min().item()})
        else:
            raise NotImplementedError

    def encode_data(self, data, h_init=None):
        batch_length = data['obs'].shape[0]
        states = defaultdict(list)

        if h_init is None:
            h_t = self._init_deter(data['obs'].shape[1])
        else:
            h_t = h_init

        for t in range(batch_length):
            obs = data['obs'][t]

            if self.config.z_dist == 'Categorical':
                post, prior = self.world_model.get_z_dists(h_t, obs)
                z_t = post.sample()
                latents = {'state': torch.cat((h_t, z_t), dim=-1),
                           'post': post.logits, 'prior': prior.logits}
            elif self.config.z_dist == 'Gaussian':
                (post, post_stats), (prior, prior_stats) = \
                    self.world_model.get_z_dists(h_t, obs)
                z_t = post.sample()
                latents = {'state': torch.cat((h_t, z_t), dim=-1),
                           'post': post_stats, 'prior': prior_stats}
            else:
                raise NotImplementedError

            for k, v in latents.items():
                states[k].append(v)
            if t < batch_length - 1:
                h_t = self.world_model.forward(h_t, z_t, data['action'][t])

        states = {k: torch.stack(v, dim=0) for k, v in states.items()}
        states['action'] = data['action']
        return states, h_t

    # -------------------------------------------------------------------------
    # Ensemble

    def _train_ensemble(self, data, states):
        self.world_model.requires_grad_(False)

        actions = data['action'][:, :self.config.ensemble_size]
        states = states['state'][:, :self.config.ensemble_size].detach()
        pred_states = self.ensemble(torch.cat((states[:-1],
                                               actions[:-1]), dim=-1))
        loss = (pred_states - states[1:]).square().mean()

        # update weights
        self.ensemble_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(),
                                       self.config.model_grad_clip)
        self.ensemble_optim.step()

        self.world_model.requires_grad_(True)
        self.logger.log('ensemble', {'loss': loss.item()})

    # -------------------------------------------------------------------------
    # Actor Critic

    def _train_actor_critic(self, states_0):
        self.world_model.requires_grad_(False)
        self.ensemble.requires_grad_(False)

        states_t = states_0
        actor = self.actor(states_t)
        act_t = actor.sample()

        # imagine rollouts
        states, actions, action_log_probs = [], [], []
        states_logits = []
        info = {'entropy': 0, 'size': 0, 'std': 0}
        for _ in range(self.config.imag_horizon):
            states_t = self.world_model.step(states_t, act_t)
            log_prob = actor.log_prob(act_t.detach())

            # for ditto
            if self.config.z_dist == 'Categorical':
                states_logits_t = self.world_model.dynamics_logits(
                    states_t[..., :self.config.h_dim])
                states_logits.append(torch.cat((
                    states_t[..., :self.config.h_dim],
                    states_logits_t), dim=-1))

            states.append(states_t)
            actions.append(act_t)
            action_log_probs.append(log_prob)

            info['entropy'] += actor.entropy().mean() / \
                self.config.imag_horizon
            info['size'] += act_t.square().mean() / \
                self.config.imag_horizon
            info['std'] += actor.stddev.mean() / \
                self.config.imag_horizon

            # sample next action
            actor = self.actor(states_t)
            act_t = actor.sample()

        # collect extra state for amp
        if self.config.amp:
            self.state_last = self.world_model.step(
                states_t, act_t).unsqueeze(0)

        self.states = torch.stack(states, dim=0)
        self.actions = torch.stack(actions, dim=0)
        self.action_log_probs = torch.stack(
            action_log_probs, dim=0).unsqueeze(-1)
        if self.config.z_dist == 'Categorical':
            self.states_logits = torch.stack(states_logits, dim=0)

        # mean distance between agent and expert actions
        act_diff = torch.norm(self.actions - self.expert_actions[1:],
                              dim=-1).mean()

        # amp training
        if self.config.amp:
            self._train_disc()

        # calculate rewards
        if self.config.Plan2Explore:
            self.rewards = self.ensemble.get_variance(
                torch.cat((states, actions), dim=-1))
        elif self.config.ditto:
            self._calculate_ditto_rewards()
        else:
            self.rewards = self.world_model.reward(states)

        self._calculate_gammas()

        # calculate losses
        policy_loss = self._calculate_policy_loss()
        value_loss = self._calculate_value_loss()
        entropy_loss = self.config.entropy_coeff * info['entropy']
        loss = policy_loss + value_loss - entropy_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                       self.config.ac_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                       self.config.ac_grad_clip)
        self.actor_optim.step()
        self.critic_optim.step()

        self._update_slow_critic()
        self.world_model.requires_grad_(True)
        self.ensemble.requires_grad_(True)

        self.logger.log('actor_info', {'act_diff': act_diff, **info})
        self.logger.log('actor_reward', {'reward': self.rewards.mean()})
        self.logger.log('actor_critic_loss',
                        {'policy_loss': policy_loss, 'value_loss': value_loss})

    def _calculate_policy_loss(self):
        self.critic.requires_grad_(False)

        value_targets = []
        if self.config.critic_update == 'hard':
            values = self.slow_critic(self.states).mode()
        elif self.config.critic_update == 'soft':
            values = self.critic(self.states).mode()
        else:
            raise NotImplementedError

        for t in reversed(range(self.config.imag_horizon - 1)):
            if t == self.config.imag_horizon - 2:
                target = values[t + 1]
            else:
                target = (1 - self.config.lam) * values[t + 1] + \
                    self.config.lam * value_targets[-1]
            value_targets.append(self.rewards[t] + self.gammas[t] * target)

        value_targets = torch.stack(value_targets[::-1], dim=0)
        offset, scale = self.reward_ema(value_targets)
        norm_targets = (value_targets - offset) / scale
        norm_base = (values[:-1] - offset) / scale
        adv = norm_targets - norm_base

        if self.config.policy_gradient == 'dynamics':
            actor_target = adv
        elif self.config.policy_gradient == 'reinforce':
            actor_target = self.action_log_probs[:-1] * adv.detach()
        else:
            raise NotImplementedError(
                f'Unknown policy gradient: {self.config.policy_gradient}')

        policy_loss = -(self.weights[:-1] * actor_target).mean()
        self.value_targets = value_targets.detach()
        self.logger.log('critic',
                        {'value_targets': self.value_targets.mean(),
                         'value_target_ema/05': self.reward_ema.values[0],
                         'value_target_ema/95': self.reward_ema.values[1]})

        self.critic.requires_grad_(True)
        return policy_loss

    def _calculate_value_loss(self):
        self.actor.requires_grad_(False)

        values = self.critic(self.states[:-1].detach())
        value_loss = -values.log_prob(self.value_targets)
        if self.config.critic_update == 'soft':
            slow_critic = self.slow_critic(self.states[:-1].detach())
            value_loss -= values.log_prob(slow_critic.mode().detach())
        value_loss = (self.weights[:-1] * value_loss.unsqueeze(-1)).mean()

        self.actor.requires_grad_(True)
        self.logger.log('critic', {'values': values.mode().mean()})
        return value_loss

    def _update_slow_critic(self):
        if self.config.critic_update == 'hard':
            if self._updates % self.config.critic_update_freq == 0:
                self.slow_critic.load_state_dict(self.critic.state_dict())
        elif self.config.critic_update == 'soft':
            mix = self.config.critic_update_fraction
            for s, d in zip(self.critic.parameters(),
                            self.slow_critic.parameters()):
                d.data = mix * s.data + (1 - mix) * d.data
        self._updates += 1

    def _calculate_ditto_rewards(self):
        if self.config.amp:
            states = torch.cat([self.states, self.state_last], dim=0)
            states = torch.cat([states[:-1, :, :self.h_dim],
                                states[1:, :, :self.h_dim]], dim=-1)
            reward = torch.maximum(
                torch.zeros_like(states).to(self.config.device),
                1 - 0.25 * (self.disc(states) - 1)**2).mean(dim=-1)
        else:
            if self.config.ditto_state == 'logits':
                states = self.states_logits
            else:
                states = self.states

            if self.config.ditto_state == 'deter':
                expert_states = self.expert_states[1:, :, :self.config.h_dim]
                states = states[..., :self.config.h_dim]
            else:
                expert_states = self.expert_states[1:]

            reward = torch.sum(expert_states * states, dim=-1)
            reward /= (torch.maximum(torch.norm(expert_states, dim=-1),
                                     torch.norm(states, dim=-1)) ** 2)

        act_diff = torch.norm(self.actions - self.expert_actions[1:], dim=-1)
        reward -= self.config.act_diff_coeff * act_diff
        self.rewards = reward.unsqueeze(-1)

    # -------------------------------------------------------------------------
    # AMP

    def _train_disc(self):
        # stack states into pairs
        expert_states = torch.cat([self.expert_states[1:-1, :, :self.h_dim],
                                   self.expert_states[2:, :, :self.h_dim]],
                                  dim=-1)
        states = torch.cat([self.states[:-1, :, :self.h_dim],
                            self.states[1:, :, :self.h_dim]],
                           dim=-1).detach()
        assert expert_states.shape == states.shape
        assert expert_states.shape == (self.config.imag_horizon - 1,
                                       self.config.ditto_batch_size,
                                       2 * self.h_dim)

        # generate labels
        expert_pred = self.disc(expert_states)
        agent_pred = self.disc(states)

        # calculate loss
        pred_loss = (expert_pred - 1).pow(2).mean() + \
            (agent_pred + 1).pow(2).mean()
        grad_penalty = self._calculate_gradient_penalty(expert_states)
        loss = pred_loss + grad_penalty

        # update
        self.disc_optim.zero_grad()
        loss.backward()
        self.disc_optim.step()

        # logging
        self.logger.log('disc', {'pred_loss': pred_loss,
                                 'grad_penalty': grad_penalty})

    def _calculate_gradient_penalty(self, expert_states):
        expert_pred = self.disc(expert_states.requires_grad_())
        grad = torch.autograd.grad(inputs=expert_states, outputs=expert_pred,
                                   grad_outputs=torch.ones_like(
                                       expert_pred).to(self.device),
                                   create_graph=True,
                                   retain_graph=True)[0]
        return torch.norm(grad).pow(2).mean()

    # -------------------------------------------------------------------------
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
            raise NotImplementedError(
                f'Unknown init_deter: {self.config.init_deter}')

    def _calculate_gammas(self):
        # weights are to softly account for episode termination
        if self.config.ditto:
            # cont model isn't use for ditto
            self.gammas = self.config.gamma * torch.ones_like(self.rewards)
            self.weights = torch.ones_like(self.gammas).detach()
        else:
            self.gammas = self.config.gamma * self.world_model.cont(
                self.states)
            self.weights = torch.cumprod(
                self.gammas, dim=0).detach()

    def set_expert_data_size(self, expert_sampler):
        self.expert_data_size = (
            expert_sampler.len_samples, expert_sampler.n_batches)

    def _get_z_dist(self):
        if self.config.z_dist == 'Categorical':
            return lambda x: common.CategoricalDist(
                x, self.config.unimix_ratio,
                dim=int(math.sqrt(self.config.z_dim)))
        elif self.config.z_dist == 'Gaussian':
            return lambda x: D.Normal(*x.split(self.config.z_dim, dim=-1))

    def kl_div(self, x, y):
        min_kl = 0
        if self.config.z_dist == 'Categorical':
            x, y = x.dist, y.dist
            min_kl = 1.0
        return torch.clip(D.kl.kl_divergence(x, y), min=min_kl)
