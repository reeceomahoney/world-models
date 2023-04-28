import unittest
import dreamer.common as common
import torch
from tests.test_ensemble import _init_config


class TestRSSM(unittest.TestCase):
    def test_step(self):
        obs_dim, act_dim = 10, 5
        config = _init_config()
        rssm = common.WorldModel(obs_dim, act_dim, config.h_dim, config.z_dim, **config.rssm)
        state = torch.randn(1, config.h_dim + config.z_dim).cuda()
        a_t = torch.randn(1, act_dim).cuda()
        x = rssm.step(state, a_t)
        self.assertEqual(x.shape, (1, config.h_dim + config.z_dim))

    def test_post(self):
        obs_dim, act_dim = 10, 5
        config = _init_config()
        rssm = common.WorldModel(obs_dim, act_dim, config.h_dim, config.z_dim, **config.rssm)
        h_t = torch.randn(1, config.h_dim).cuda()
        o_t = torch.randn(1, obs_dim).cuda()
        post = rssm.encode(h_t, o_t)
        self.assertEqual(post.shape, (1, config.z_dim))

    def test_predict(self):
        obs_dim, act_dim = 10, 5
        config = _init_config()
        rssm = common.WorldModel(obs_dim, act_dim, config.h_dim, config.z_dim, **config.rssm)
        h_t = torch.randn(1, config.h_dim).cuda()
        o_t = torch.randn(1, obs_dim).cuda()
        a_t = torch.randn(1, act_dim).cuda()
        preds, h_t1, _ = rssm.predict(h_t, o_t, a_t)
        self.assertEqual(preds[0].shape, (1, obs_dim))
        self.assertEqual(preds[1].shape, (1, 1))
        self.assertEqual(preds[2].shape, (1, 1))

    def test_log_probs(self):
        torch.manual_seed(0)
        obs_dim, act_dim = 10, 5
        config = _init_config()
        rssm = common.WorldModel(obs_dim, act_dim, config.h_dim, config.z_dim, **config.rssm)

        h_t = torch.randn(1, config.h_dim).cuda()
        o_t = torch.randn(1, obs_dim).cuda()
        r_t = torch.randn(1, 1).cuda()
        g_t = torch.randint(0, 2, (1, 1)).cuda()
        a_t = torch.randn(1, act_dim).cuda()

        log_probs, state = rssm.log_probs(h_t, o_t, r_t, g_t, a_t)

        self.assertEqual(len(log_probs), 3)


if __name__ == '__main__':
    unittest.main()
