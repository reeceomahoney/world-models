import os
import unittest

import numpy as np
from ruamel.yaml import YAML
import torch

import dreamer.common as common
from dreamer.agent import Agent


def _init_agent():
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../dreamer/config.yaml', 'r') as f:
        config_dict = YAML().load(f)['default']
    config = common.Config(config_dict)
    obs_dim, act_dim, act_range = 10, 5, 1
    return Agent(obs_dim, act_dim, act_range, config)


def _get_obs():
    return np.random.rand(1, 10), np.random.rand(1, 1), np.random.rand(1, 1), np.random.rand(1, 5)


def _init_with_obs():
    torch.manual_seed(0)
    agent = _init_agent()
    obs, reward, done, action = _get_obs()
    for _ in range(49):
        agent.store(obs, reward, np.zeros((1, 1)), action)
    agent.store(obs, reward, np.ones((1, 1)), action)
    return agent


class TestTrainer(unittest.TestCase):
    def test_store(self):
        obs, reward, done, action = _get_obs()
        trainer = _init_agent()
        trainer.store(obs, reward, done, action)
        self.assertEqual(len(trainer.replay), 1)

    def test_step(self):
        obs, reward, done, action = _get_obs()
        trainer = _init_agent()
        h_t = torch.randn(1, trainer.config.h_dim).cuda()
        h_t1, a_t = trainer.step(h_t, obs)
        self.assertEqual(h_t1.shape, (1, trainer.config.h_dim))
        self.assertEqual(a_t.shape, (1, 5))

    def test_predict(self):
        obs = np.random.rand(1, 10)
        act = torch.randn(1, 5).cuda()
        agent = _init_agent()
        h_t = torch.randn(1, agent.config.h_dim).cuda()

        # test with policy
        preds, h_t1, a_t = agent.step(h_t, obs)
        self.assertEqual(preds[0].shape, (1, 10))
        self.assertEqual(preds[1].shape, (1, 1))
        self.assertEqual(preds[2].shape, (1, 1))
        self.assertEqual(h_t1.shape, (1, agent.config.h_dim))
        self.assertEqual(a_t.shape, (1, 5))

        # test with action
        preds, h_t1, _ = agent.step(h_t, obs, act)
        self.assertEqual(preds[0].shape, (1, 10))
        self.assertEqual(preds[1].shape, (1, 1))
        self.assertEqual(preds[2].shape, (1, 1))
        self.assertEqual(h_t1.shape, (1, agent.config.h_dim))

    def test_train_world_model(self):
        agent = _init_with_obs()
        info = agent.train_world_model()
        self.assertEqual('recon_loss' in info, True)
        self.assertEqual('kl_loss' in info, True)

    def test_train_ensemble(self):
        agent = _init_with_obs()
        info = agent._train_ensemble()
        self.assertEqual('ensemble_loss' in info, True)

    def test_train_actor_critic(self):
        agent = _init_with_obs()
        agent.train_world_model()
        info = agent._train_actor_critic(False)
        self.assertEqual(len(info), 4)

    def test_train(self):
        agent = _init_with_obs()
        info = agent.train_step(explore=True)
        self.assertIsInstance(info, dict)
        info = agent.train_step(explore=False)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
