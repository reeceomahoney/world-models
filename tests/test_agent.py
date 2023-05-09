import os
import argparse
import unittest

import torch

from .test_storage import fill_replay
import world_models.common as common
from world_models.agent import Agent


def setup(ditto=False):
    args = argparse.ArgumentParser()
    args.env = 'raisim'
    args.ditto = ditto

    current_dir = os.path.dirname(os.path.realpath(__file__))
    config = common.init_config(current_dir + '/config.yaml', args)[0]
    obs_dim, act_dim = 36, 12
    agent = Agent(obs_dim, act_dim, 1, config)
    replay = common.ReplayBuffer(config, {'obs': obs_dim, 'reward': 1, 'cont': 1, 'action': act_dim})
    return config, agent, replay, obs_dim, act_dim


class TestTrainer(unittest.TestCase):
    def test_call(self):
        config, agent, replay, obs_dim, act_dim = setup()
        h_t = torch.randn(1, config.h_dim).to(config.device)
        obs = torch.randn(1, obs_dim).to(config.device)
        h_t, action = agent(h_t, obs)
        self.assertEqual(h_t.shape, (1, config.h_dim))
        self.assertEqual(action.shape, (1, act_dim))

    def test_predict(self):
        config, agent, replay, obs_dim, act_dim = setup()
        h_t = torch.randn(1, config.h_dim).to(config.device)
        obs = torch.randn(1, obs_dim).to(config.device)
        preds, h_t, action = agent.predict(h_t, obs)

        self.assertEqual(h_t.shape, (1, config.h_dim))
        self.assertEqual(action.shape, (1, act_dim))
        self.assertEqual(preds[0].shape, (1, obs_dim))
        self.assertEqual(preds[1].shape, (1, 1))
        self.assertEqual(preds[2].shape, (1, 1))

    def test_train_step(self):
        config, agent, replay, obs_dim, act_dim = setup()
        replay = fill_replay(replay, 1000, config, obs_dim, act_dim)
        info = agent.train_step(0, replay, True)
        self.assertEqual('pred_loss' in info, True)
        self.assertEqual('policy_loss' in info, True)

    def test_ditto_step(self):
        config, agent, replay, obs_dim, act_dim = setup(ditto=True)
        state_replay = common.ReplayBuffer(config, {'state': config.h_dim + config.z_dim})
        replay = fill_replay(replay, 1000, config, obs_dim, act_dim)
        states = agent.encode_expert_data(replay)[-1]
        state_replay.store_all_from_tensors(states)

        info = agent.ditto_step(state_replay)
        self.assertEqual('policy_loss' in info, True)


if __name__ == '__main__':
    unittest.main()
