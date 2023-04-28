import unittest
import dreamer.common as common
import torch


class TestNormaliser(unittest.TestCase):
    def test_update_stats(self):
        obs_dim, envs = 4, 2
        normaliser = common.Normaliser(obs_dim, envs)
        x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
        normaliser.update_stats(x)
        self.assertTrue(torch.allclose(normaliser.mean, torch.Tensor([3, 4, 5, 6]).cuda()))
        self.assertTrue(torch.allclose(normaliser.std, torch.Tensor([2, 2, 2, 2]).cuda()))

    def test_normalise(self):
        obs_dim, envs = 4, 2
        normaliser = common.Normaliser(obs_dim, envs)
        x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
        normaliser.update_stats(x)
        obs = normaliser.normalise(x)
        self.assertTrue(torch.allclose(obs, torch.Tensor([[-1, -1, -1, -1], [1, 1, 1, 1]]).cuda()))

    def test_unnormalise(self):
        obs_dim, envs = 4, 2
        normaliser = common.Normaliser(obs_dim, envs)
        x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
        normaliser.update_stats(x)
        obs = normaliser.unnormalise(torch.Tensor([[-1, -1, -1, -1], [1, 1, 1, 1]]).cuda())
        self.assertTrue(torch.allclose(obs, x))


if __name__ == '__main__':
    unittest.main()
