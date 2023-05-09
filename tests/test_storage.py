import argparse
import unittest

import torch

from world_models import common


def setup():
    args = argparse.ArgumentParser()
    args.env = 'raisim'
    args.ditto = False
    config = common.init_config('config.yaml', args)[0]
    obs_dim, act_dim = 36, 12
    replay = common.ReplayBuffer(config, {'obs': obs_dim, 'reward': 1, 'cont': 1, 'action': act_dim})
    return config, replay, obs_dim, act_dim


def fill_replay(replay, n, config, obs_dim, act_dim):
    step = 0
    for _ in range(n):
        replay.store({'obs': torch.randn(1, obs_dim).to(config.device),
                      'reward': torch.randn(1, 1).to(config.device),
                      'cont': torch.zeros(1, 1).to(config.device),
                      'action': torch.randn(1, act_dim).to(config.device)})
        step += 1
        if step >= config.time_limit:
            replay.add_episode()
            step = 0

    return replay


class TestReplay(unittest.TestCase):
    def test_store(self):
        config, replay, obs_dim, act_dim = setup()
        replay = fill_replay(replay, 1200, config, obs_dim, act_dim)
        self.assertEqual(len(replay), 1000)
        self.assertEqual(len(replay.episodes), 10)

    def test_sample(self):
        config, replay, obs_dim, act_dim = setup()
        replay = fill_replay(replay, 1000, config, obs_dim, act_dim)

        batch_size = 32
        for _ in range(2):
            data = replay.sample(batch_size)
            self.assertEqual(data['obs'].shape, (64, batch_size, obs_dim))
            self.assertEqual(data['reward'].shape, (64, batch_size, 1))
            self.assertEqual(data['cont'].shape, (64, batch_size, 1))
            self.assertEqual(data['action'].shape, (64, batch_size, act_dim))

            config.time_limit = config.batch_length  # for second pass


if __name__ == '__main__':
    unittest.main()
