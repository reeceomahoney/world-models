import os

import numpy as np
import torch
import gymnasium as gym
import world_models.raisim_gym as raisim_gym
from ruamel.yaml import dump, RoundTripDumper

from .utils import symlog


def to_np(x):
    return x.detach().cpu().numpy()


def get_driver(config, config_dict=None):
    if config.env_name == 'raisim':
        return RaisimDriver(config, config_dict)
    else:
        return GymDriver(config)


class DriverBase:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self._env = None

    def _init_deter(self):
        if self.config.init_deter == 'zero':
            return torch.zeros((self.config.num_envs, self.config.h_dim)).to(
                self.config.device)
        elif self.config.init_deter == 'normal':
            return 0.01 * torch.randn((
                self.config.num_envs, self.config.h_dim)).to(
                    self.config.device)

    def _to_ten(self, x):
        return torch.tensor(x, dtype=torch.float32).to(self.config.device)


class GymDriver(DriverBase):
    def __init__(self, config, render=False):
        super(GymDriver, self).__init__(config)
        self._config = config
        self._render = render
        self._make_env()

    def __call__(self, action):
        self.step += 1
        obs, reward, done = None, None, None
        for _ in range(self.config.action_repeat):
            obs, reward, done = self._env.step(action)[:3]
        return obs, reward, done

    def reset(self):
        if self._config.record:
            self._env.close()
            self._make_env()
        self.step = 0
        h_t = self._init_deter()
        obs = self._env.reset()[0]
        action = self._env.action_space.sample()
        return obs, h_t, action

    def env_info(self):
        obs_dim = self._env.observation_space.shape[-1]
        act_dim = self._env.action_space.shape[-1]
        act_max = self._env.action_space.high[0][0]
        return obs_dim, act_dim, act_max

    def close(self):
        self._env.close()

    def _make_env(self):
        if self._config.record:
            video_path = os.path.dirname(os.path.realpath(__file__)) + \
                f'/../../logs/{self._config.env_name}/videos'
            self._env = gym.vector.make(
                self._config.env_name,
                render_mode='rgb_array',
                num_envs=self._config.num_envs,
                wrappers=lambda x: self._wrapper(x, video_path))
        elif self._render:
            self._env = gym.vector.make(
                self._config.env_name,
                render_mode='human',
                num_envs=self._config.num_envs)

    def _wrapper(self, x, video_path):
        return gym.wrappers.RecordVideo(x, video_path,
                                        episode_trigger=lambda y: True)


class RaisimDriver(DriverBase):
    def __init__(self, config, config_dict):
        super(RaisimDriver, self).__init__(config)
        self._raisim_config = config_dict
        rsc_path = os.path.dirname(os.path.realpath(__file__)) + \
            '/../raisim_gym/rsc'

        self._env = raisim_gym.VecEnv(raisim_gym.RaisimGymEnv(
            rsc_path, dump(self._raisim_config, Dumper=RoundTripDumper)),
                                      normalize_ob=False)
        self._env.turn_off_visualization()

        self.expert_data = None
        self.start_idx = None
        self.eps_idx = None

    def __call__(self, action):
        self.step += 1
        obs = symlog(self._to_ten(self._env.observe()))
        reward, done = self._env.step(to_np(action))
        return obs, self._to_ten(reward).unsqueeze(-1), \
            self._to_ten(done).unsqueeze(-1)

    def reset(self):
        self.step = 0
        if self.config.expert_init_state:
            init_data = self.sample_expert_data()
            self._env.expert_reset(init_data)
        else:
            self._env.reset()
        h_t = self._init_deter()
        obs = self._to_ten(self._env.observe())
        action = torch.randn(self.config.num_envs, self._env.num_acts).to(
            self.config.device)
        return obs, h_t, action

    def load_expert_data(self, expert_data):
        self.expert_data = expert_data

    def sample_expert_data(self):
        self.start_idx = torch.randint(
            0, self.expert_data.shape[0] - self.config.eval_steps, (1,))
        self.eps_idx = torch.randint(
            0, self.expert_data.shape[1], (1,))
        sample = self.expert_data[
            self.start_idx:self.start_idx + self.config.eval_steps,
            self.eps_idx]
        return np.squeeze(to_np(sample), axis=1)

    def env_info(self):
        return self._env.num_obs, self._env.num_acts, self.config.action_clip

    def turn_on_visualization(self):
        self._env.turn_on_visualization()

    def turn_off_visualization(self):
        self._env.turn_off_visualization()

    def get_reward_info(self):
        return self._env.get_reward_info()

    def set_target(self, target):
        self._env.set_target(target)

    def get_init_row(self):
        return self._env.get_init_row()
