import math

import torch
import torch.distributions as D
import torch.nn as nn

from .distributions import TruncatedNormal, SymlogGaussian, \
    TwoHotDistSymlog, CategoricalDist
from .utils import symexp, act_case


class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, act, device,
                 init_zero=False):
        super(BaseMLP, self).__init__()

        layers = []
        dims = [in_dim, *hidden_layers, out_dim]
        act = act_case(act)
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.LayerNorm(dims[idx + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.architecture = nn.Sequential(*layers).to(device)
        self.out_dim = out_dim

        if init_zero:
            for layer in self.architecture:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()

    def __call__(self, x):
        return self.architecture(x)


class Actor(BaseMLP):
    def __init__(self, act_dim, act_range, config):
        super(Actor, self).__init__(config.h_dim + config.z_dim, 2 * act_dim,
                                    config.layers, config.act, config.device)

        self._act_dim = act_dim
        self._act_range = act_range
        self._init_std = config.init_std
        self._max_std = config.max_std
        self._min_std = config.min_std

    def __call__(self, x):
        x = self.architecture(x)
        mean, std = torch.split(x, self._act_dim, dim=-1)
        mean = torch.tanh(mean)
        std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + \
            self._min_std
        dist = TruncatedNormal(mean, std, -self._act_range, self._act_range)
        return D.Independent(dist, 1)


class Decoder(BaseMLP):
    def __init__(self, in_dim, out_dim, layers, act, device='cuda'):
        super(Decoder, self).__init__(in_dim, out_dim, layers, act, device)

    def __call__(self, x):
        dist = D.Normal(symexp(self.architecture(x)), 1)
        return D.Independent(dist, 1)


class MultivariateGaussianMLP(BaseMLP):
    def __init__(self, in_dim, config):
        super(MultivariateGaussianMLP, self).__init__(
            in_dim, 2 * config.z_dim, config.layers, config.act, config.device)

        self.z_dim = config.z_dim
        self.init_std = config.init_std
        self.max_std = config.max_std
        self.min_std = config.min_std

    def __call__(self, x):
        x = self.architecture(x)
        mean, std = torch.split(x, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + \
            self.min_std
        dist = D.Normal(mean, std)
        return D.Independent(dist, 1), torch.cat([mean, std], dim=-1)


class GaussianMLP(BaseMLP):
    def __init__(self, config):
        super(GaussianMLP, self).__init__(
            config.h_dim + config.z_dim, 1, config.layers, config.act,
            config.device)

    def __call__(self, x):
        dist = SymlogGaussian(self.architecture(x), 1)
        return D.Independent(dist, 1)


class TwoHotSymlogMLP(BaseMLP):
    def __init__(self, config):
        super(TwoHotSymlogMLP, self).__init__(
            config.h_dim + config.z_dim, 255, config.layers, config.act,
            config.device, init_zero=config.init_zero)
        self.device = config.device

    def __call__(self, x):
        return TwoHotDistSymlog(self.architecture(x), device=self.device)


class RecurrentModel(nn.Module):
    def __init__(self, in_dim, hidden_state_dim, device):
        super(RecurrentModel, self).__init__()

        self.architecture = nn.GRUCell(
            input_size=in_dim, hidden_size=hidden_state_dim)
        self.architecture.to(device)

    def __call__(self, x, h):
        return self.architecture(x, h)


class CategoricalMLP(BaseMLP):
    def __init__(self, in_dim, out_dim, config, device):
        super(CategoricalMLP, self).__init__(in_dim, out_dim, config.layers,
                                             config.act, device)
        self.unimix_ratio = config.unimix_ratio
        self.dim = int(math.sqrt(out_dim))

    def __call__(self, x):
        return CategoricalDist(self.architecture(x),
                               self.unimix_ratio, self.dim)


class BernoulliMLP(BaseMLP):
    def __init__(self, in_dim, out_dim, layers, act, device='cuda'):
        super(BernoulliMLP, self).__init__(in_dim, out_dim, layers, act,
                                           device)

    def __call__(self, x):
        logits = self.architecture(x)
        dist = D.Bernoulli(logits=logits)
        return D.Independent(dist, 1)


class Ensemble(nn.Module):
    def __init__(self, act_dim, config):
        super(Ensemble, self).__init__()
        in_dim = config.h_dim + config.z_dim + act_dim
        out_dim = config.h_dim + config.z_dim
        size = config.ensemble_size

        self.models = nn.ModuleList([
            Decoder(in_dim, out_dim, config.layers, config.act, config.device)
            for _ in range(size)])
        self.size = size
        self._explore_coeff = config.explore_coeff

    def __call__(self, x):
        return torch.stack([self.models[i](x).mode for i in range(self.size)])

    def get_variance(self, x):
        return self._explore_coeff * self(x).var(dim=0).mean(
            dim=-1, keepdim=True)
