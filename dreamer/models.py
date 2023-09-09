import math

import torch
import torch.distributions as D
import torch.nn as nn

from .distributions import (
    TruncatedNormal,
    SymlogGaussian,
    TwoHotDistSymlog,
    CategoricalDist,
    DecoderDist,
)
from .utils import symexp, act_case


class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, act, device, init_zero=False):
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
        super(Actor, self).__init__(
            config.h_dim + config.z_dim,
            2 * act_dim,
            config.layers,
            config.act,
            config.device,
        )

        self._act_dim = act_dim
        self._act_range = act_range
        self._init_std = config.init_std
        self._max_std = config.max_std
        self._min_std = config.min_std

    def __call__(self, x):
        x = self.architecture(x)
        mean, std = torch.split(x, self._act_dim, dim=-1)
        mean = torch.tanh(mean)
        std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std


class CategoricalActor(BaseMLP):
    def __init__(self, act_dim, config):
        super(CategoricalActor, self).__init__(
            config.h_dim + config.z_dim,
            act_dim,
            config.layers,
            config.act,
            config.device,
        )

    def __call__(self, x):
        return torchd.Bernoulli(logits=self.architecture(x))


class Decoder(BaseMLP):
    def __init__(self, in_dim, out_dim, layers, act, device="cuda"):
        super(Decoder, self).__init__(in_dim, out_dim, layers, act, device)

    def __call__(self, x):
        return DecoderDist(symexp(self.architecture(x)))


class MultivariateGaussianMLP(BaseMLP):
    def __init__(self, in_dim, config):
        super(MultivariateGaussianMLP, self).__init__(
            in_dim, 2 * config.z_dim, config.layers, config.act, config.device
        )

        self.z_dim = config.z_dim
        self.init_std = config.init_std
        self.max_std = config.max_std
        self.min_std = config.min_std

    def __call__(self, x):
        x = self.architecture(x)
        mean, std = torch.split(x, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        dist = D.Normal(mean, std)
        return D.Independent(dist, 1), torch.cat([mean, std], dim=-1)


class GaussianMLP(BaseMLP):
    def __init__(self, config):
        super(GaussianMLP, self).__init__(
            config.h_dim + config.z_dim, 1, config.layers, config.act, config.device
        )

    def __call__(self, x):
        dist = SymlogGaussian(self.architecture(x), 1)
        return D.Independent(dist, 1)


class TwoHotSymlogMLP(BaseMLP):
    def __init__(self, config):
        super(TwoHotSymlogMLP, self).__init__(
            config.h_dim + config.z_dim,
            255,
            config.layers,
            config.act,
            config.device,
            init_zero=config.init_zero,
        )
        self.device = config.device

    def __call__(self, x):
        return TwoHotDistSymlog(self.architecture(x), device=self.device)


class RecurrentModel(nn.Module):
    def __init__(self, in_dim, hidden_state_dim, device):
        super(RecurrentModel, self).__init__()

        self.architecture = nn.GRUCell(input_size=in_dim, hidden_size=hidden_state_dim)
        self.architecture.to(device)

    def __call__(self, x, h):
        return self.architecture(x, h)


class CategoricalMLP(BaseMLP):
    def __init__(self, in_dim, out_dim, config, device):
        super(CategoricalMLP, self).__init__(
            in_dim, out_dim, config.layers, config.act, device
        )
        self.unimix_ratio = config.unimix_ratio
        self.dim = int(math.sqrt(out_dim))

    def __call__(self, x):
        return CategoricalDist(self.architecture(x), self.unimix_ratio, self.dim)


class BernoulliMLP(BaseMLP):
    def __init__(self, in_dim, out_dim, layers, act, device="cuda"):
        super(BernoulliMLP, self).__init__(in_dim, out_dim, layers, act, device)

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

        self.models = nn.ModuleList(
            [
                Decoder(in_dim, out_dim, config.layers, config.act, config.device)
                for _ in range(size)
            ]
        )
        self.size = size
        self._explore_coeff = config.explore_coeff

    def __call__(self, x):
        return torch.stack([self.models[i](x).mode for i in range(self.size)])

    def get_variance(self, x):
        return self._explore_coeff * self(x).var(dim=0).mean(dim=-1, keepdim=True)


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(WorldModel, self).__init__()
        self.config = config
        layers = config.layers
        act = config.act
        device = config.device
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim

        # rssm core
        self._recurrent_model = RecurrentModel(self.z_dim + act_dim, self.h_dim, device)
        if config.z_dist == "Categorical":
            self._encoder = CategoricalMLP(
                self.h_dim + obs_dim, self.z_dim, config, device
            )
            self._dynamics = CategoricalMLP(self.h_dim, self.z_dim, config, device)
        elif config.z_dist == "Gaussian":
            self._encoder = MultivariateGaussianMLP(self.h_dim + obs_dim, config)
            self._dynamics = MultivariateGaussianMLP(self.h_dim, config)
        else:
            raise NotImplementedError("Unknown z_dist")

        # prediction
        self._decoder = Decoder(self.h_dim + self.z_dim, obs_dim, layers, act, device)
        self._reward_model = TwoHotSymlogMLP(config)
        self._cont_model = BernoulliMLP(self.h_dim + self.z_dim, 1, layers, act, device)

    def forward(self, h_t, z_t, action):
        # step recurrent state
        return self._recurrent_model(torch.cat((z_t, action), dim=-1), h_t)

    def dynamics(self, h_t):
        if self.config.z_dist == "Categorical":
            return self._dynamics(h_t).sample()
        elif self.config.z_dist == "Gaussian":
            return self._dynamics(h_t)[0].sample()

    def dynamics_logits(self, h_t):
        return self._dynamics(h_t).logits

    def encode(self, h_t, obs):
        if self.config.z_dist == "Categorical":
            return self._encoder(torch.cat((h_t, obs), dim=-1)).sample()
        elif self.config.z_dist == "Gaussian":
            return self._encoder(torch.cat((h_t, obs), dim=-1))[0].sample()

    def decode(self, state):
        return self._decoder(state).mode()

    def step(self, state, action):
        # step latent state in imagination
        h_t1 = self.forward(state[..., : self.h_dim], state[..., self.h_dim :], action)
        z_t1 = self.dynamics(h_t1)
        return torch.cat((h_t1, z_t1), dim=-1)

    def log_probs(self, data, states):
        log_probs = []
        if "obs" in data:
            log_probs.append(self._decoder(states).log_prob(data["obs"]))
        if "reward" in data:
            log_probs.append(self._reward_model(states).log_prob(data["reward"]))
        if "cont" in data:
            log_probs.append(self._cont_model(states).log_prob(data["cont"]))
        return torch.stack(log_probs)

    def get_z_dists(self, h_t, obs):
        return self._encoder(torch.cat((h_t, obs), dim=-1)), self._dynamics(h_t)

    def reward(self, state):
        return self._reward_model(state).mode()

    def cont(self, state):
        return self._cont_model(state).sample()

    def predict(self, state):
        obs_1 = symexp(self._decoder(state).mode())
        reward_1 = self._reward_model(state).mode()
        cont_1 = self._cont_model(state).sample()
        return obs_1, reward_1, cont_1
