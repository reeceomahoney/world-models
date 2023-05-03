import unittest

import torch

from world_models import common


class TestReplay(unittest.TestCase):
    def test_add_step(self):
        replay = common.DreamerBuffer(1, 100, 200, 10)
        for _ in range(110):
            replay.add_step({'obs': torch.randn(1, 1),
                             'reward': torch.randn(1, 1),
                             'done': torch.zeros(1, 1),
                             'action': torch.randn(1, 1)})
        self.assertEqual(len(replay), 100)

    def test_add_episode(self):
        replay = common.DreamerBuffer(1, 100, 50, 10)
        for _ in range(110):
            replay.add_step({'obs': torch.randn(1, 1),
                             'reward': torch.randn(1, 1),
                             'done': torch.zeros(1, 1),
                             'action': torch.randn(1, 1)})
        self.assertEqual(len(replay), 60)
        self.assertEqual(len(replay.episodes), 1)
        self.assertEqual(len(replay.steps['obs']), 10)

        for _ in range(40):
            replay.add_step({'obs': torch.randn(1, 1),
                             'reward': torch.randn(1, 1),
                             'done': torch.zeros(1, 1),
                             'action': torch.randn(1, 1)})
        self.assertEqual(len(replay), 100)
        self.assertEqual(len(replay.episodes), 2)
        self.assertEqual(len(replay.steps['obs']), 0)

    def test_add_episode_done(self):
        replay = common.DreamerBuffer(1, 100, 50, 10)
        for _ in range(29):
            replay.add_step({'obs': torch.randn(1, 1),
                             'reward': torch.randn(1, 1),
                             'done': torch.zeros(1, 1),
                             'action': torch.randn(1, 1)})
        replay.add_step({'obs': torch.randn(1, 1),
                         'reward': torch.randn(1, 1),
                         'done': torch.ones(1, 1),
                         'action': torch.randn(1, 1)})
        self.assertEqual(len(replay), 30)
        self.assertEqual(len(replay.episodes[0]['obs']), 30)

        replay.add_step({'obs': torch.randn(1, 1),
                         'reward': torch.randn(1, 1),
                         'done': torch.ones(1, 1),
                         'action': torch.randn(1, 1)})
        self.assertEqual(len(replay), 30)

    def test_sample(self):
        replay = common.DreamerBuffer(1, 100, 50, 10)
        for _ in range(100):
            replay.add_step({'obs': torch.randn(1, 1),
                             'reward': torch.randn(1, 1),
                             'done': torch.zeros(1, 1),
                             'action': torch.randn(1, 1)})

        samples = replay.sample(15)
        self.assertEqual(len(samples['obs']), 10)


if __name__ == '__main__':
    unittest.main()
