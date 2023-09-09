import math

import numpy as np
import torch
import torch.distributions as torchd
import torch.nn as nn
import torch.nn.functional as F

from . import distributions as dist
from . import utils


class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, act, device, init_zero=False):
        super(BaseMLP, self).__init__()

        layers = []
        dims = [in_dim, *hidden_layers, out_dim]
        act = utils.act_case(act)
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
        std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
        return torchd.Independent(dist.TanhNormal(mean, std, self._act_range), 1)


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
        # return DecoderDist(self.architecture(x))
        dist = torchd.Normal(self.architecture(x), 1)
        return torchd.Independent(dist, 1)


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
        dist = torchd.Normal(mean, std)
        return torchd.Independent(dist, 1), torch.cat([mean, std], dim=-1)


class GaussianMLP(BaseMLP):
    def __init__(self, config):
        super(GaussianMLP, self).__init__(
            config.h_dim + config.z_dim, 1, config.layers, config.act, config.device
        )

    def __call__(self, x):
        return torchd.Independent(dist.SymlogGaussian(self.architecture(x), 1), 1)


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
        return dist.TwoHotDistSymlog(self.architecture(x), device=self.device)


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
        return dist.CategoricalDist(self.architecture(x), self.unimix_ratio, self.dim)


class BernoulliMLP(BaseMLP):
    def __init__(self, in_dim, out_dim, layers, act, device="cuda"):
        super(BernoulliMLP, self).__init__(in_dim, out_dim, layers, act, device)

    def __call__(self, x):
        logits = self.architecture(x)
        dist = torchd.Bernoulli(logits=logits)
        return torchd.Independent(dist, 1)


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
            # self._encoder = CategoricalMLP(
            #     self.h_dim + obs_dim, self.z_dim, config, device
            # )
            self._encoder = ConvEncoder((64, 64, 3))
            self._post_dynamics = CategoricalMLP(
                self.h_dim + self._encoder.outdim, self.z_dim, config, device
            )
            self._dynamics = CategoricalMLP(self.h_dim, self.z_dim, config, device)
        elif config.z_dist == "Gaussian":
            self._encoder = MultivariateGaussianMLP(self.h_dim + obs_dim, config)
            self._dynamics = MultivariateGaussianMLP(self.h_dim, config)
        else:
            raise NotImplementedError("Unknown z_dist")

        # prediction
        # self._decoder = Decoder(self.h_dim + self.z_dim, obs_dim, layers, act, device)
        self._decoder = ConvDecoder(self.h_dim + self.z_dim)
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
            feat = self._encoder(obs)
            return self._post_dynamics(torch.cat((h_t, feat), dim=-1)).sample()
        elif self.config.z_dist == "Gaussian":
            return self._encoder(torch.cat((h_t, obs), dim=-1))[0].sample()

    def decode(self, state):
        return self._decoder(state).mode

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
        feat = self._encoder(obs)
        return self._post_dynamics(torch.cat((h_t, feat), dim=-1)), self._dynamics(h_t)

    def reward(self, state):
        return self._reward_model(state).mode()

    def cont(self, state):
        return self._cont_model(state).sample()

    def predict(self, state):
        obs_1 = self._decoder(state).mode
        reward_1 = self._reward_model(state).mode()
        cont_1 = self._cont_model(state).sample()
        return obs_1, reward_1, cont_1


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm="LayerNorm",
        kernel_size=4,
        minres=4,
        device="cuda",
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers).to(device)
        self.layers.apply(utils.weight_init)

    def forward(self, obs):
        # (time, batch, h, w, ch) -> (time * batch, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (time * batch, h, w, ch) -> (time * batch, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (time * batch, ...) -> (time * batch, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (time * batch, -1) -> (time, batch, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act="SiLU",
        norm="LayerNorm",
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        device="cuda",
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size).to(device)
        self._linear_layer.apply(utils.weight_init)
        in_dim = self._embed_size // (minres**2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
            bias = False
            initializer = utils.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = utils.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers).to(device)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        if len(features.shape) == 2:
            test = True
            features = features.unsqueeze(0)

        x = self._linear_layer(features)
        # (time, batch, -1) -> (time * batch, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (time, batch, -1) -> (time * batch, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (time * batch, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (time * batch, ch, h, w) -> (time * batch, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5

        if test:
            mean = mean.squeeze(0)
        return torchd.Independent(torchd.Normal(mean, 1), 3)


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        dims = [in_dim, *hidden_layers]
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(norm(dims[idx + 1], eps=1e-03))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers).to(device)
        self.layers.apply(utils.weight_init)

        self.mean_layer = nn.Linear(dims[-1], out_dim).to(device)
        self.mean_layer.apply(utils.uniform_weight_init(outscale))
        if self._std == "learned":
            self.std_layer = nn.Linear(dims[-1], out_dim).to(device)
            self.std_layer.apply(utils.uniform_weight_init(outscale))

        # TODO: don't make std layer for categorical

    def forward(self, features):
        x = features
        if self._symlog_inputs:
            x = utils.symlog(x)
        out = self.layers(x)

        mean = self.mean_layer(out)
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        return self.dist(self._dist, mean, std)

    def dist(self, distr, mean, std):
        if distr == "normal":
            return torchd.Independent(torchd.Normal(mean, std), 1)
        if distr == "binary":
            return torchd.Independent(torchd.Bernoulli(logits=mean), 1)
        if distr == "categorical":
            return dist.CategoricalDist(mean, unimix_ratio=0.01, dim=32)
        if distr == "symlog_disc":
            return dist.DiscDist(logits=mean, device=self._device)
        if distr == "symlog_mse":
            return dist.SymlogDist(mean)
        if distr is None:
            return mean
        raise NotImplementedError(dist)
