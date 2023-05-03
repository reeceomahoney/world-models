import unittest
import os
from ruamel.yaml import YAML
import dreamer.common as common
import torch


def _init_config():
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../dreamer/config.yaml', 'r') as f:
        config_dict = YAML().load(f)['default']
    config = common.Config(config_dict)
    return config


class TestEnsemble(unittest.TestCase):
    def test_forward(self):
        obs_dim, act_dim = 10, 5
        config = _init_config()
        ensemble = common.Ensemble(obs_dim, act_dim, config.h_dim, config.z_dim, **config.ensemble)
        x = torch.randn(config.ensemble['size'], config.h_dim + config.z_dim + act_dim).cuda()
        sample = ensemble.forward(x)
        self.assertEqual(sample.shape, (config.ensemble['size'], obs_dim))

    def test_get_variance(self):
        obs_dim, act_dim = 10, 5
        config = _init_config()
        ensemble = common.Ensemble(obs_dim, act_dim, config.h_dim, config.z_dim, **config.ensemble)
        x = torch.randn(config.ensemble['size'], config.h_dim + config.z_dim + act_dim).cuda()
        var = ensemble.get_variance(x)
        self.assertEqual(var.shape, (config.ensemble['size'], 1))


if __name__ == '__main__':
    unittest.main()
