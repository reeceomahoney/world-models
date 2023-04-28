import unittest
import torch
import dreamer.common as common


class TestModels(unittest.TestCase):
    def test_base_mlp(self):
        mlp = common.BaseMLP(10, 5, [10, 10], 'silu')
        self.assertEqual(mlp(torch.zeros(1, 10).cuda()).shape, (1, 5))

    def test_init_zero(self):
        mlp = common.BaseMLP(10, 5, [10, 10], 'silu', init_zero=True)
        self.assertEqual(mlp.architecture[0].bias.data.sum(), 0)
        self.assertEqual(mlp.architecture[1].bias.data.sum(), 0)

    def test_actor(self):
        actor = common.Actor(10, 5, [10, 10], 'silu', 1, 1, 1, 0.1)
        actor_dist = actor(torch.zeros(1, 10).cuda())
        self.assertEqual(actor_dist.sample().shape, (1, 5))
        self.assertEqual(actor_dist.log_prob(torch.zeros(1, 5).cuda()).shape, (1,))
        self.assertEqual(actor_dist.entropy().shape, (1,))


if __name__ == '__main__':
    unittest.main()
