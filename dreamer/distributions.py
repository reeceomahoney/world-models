from abc import ABC

import torch
import torch.distributions as torchd
import torch.nn.functional as F

from .utils import symlog, symexp


class TruncatedNormal(torchd.Normal, ABC):
    def __init__(self, mean, std, low, high):
        super(TruncatedNormal, self).__init__(mean, std)
        self.low = low
        self.high = high

    def sample(self, sample_shape=()):
        return torch.clip(super().rsample(sample_shape), self.low, self.high)

    def log_prob(self, x):
        return super().log_prob(x)


class TanhNormal(torchd.Normal, ABC):
    def __init__(self, mean, std, act_max):
        super(TanhNormal, self).__init__(mean, std)
        self.act_max = act_max

    def sample(self, sample_shape=()):
        return self.act_max * torch.tanh(super().rsample(sample_shape))

    def log_prob(self, x):
        return super().log_prob(torch.atanh(x / self.act_max))

    @property
    def mode(self):
        return self.act_max * torch.tanh(super().mode)


class SymlogGaussian(torchd.Normal, ABC):
    def __init__(self, mean, std):
        super(SymlogGaussian, self).__init__(symlog(mean), std)

    def mode(self):
        return symexp(super().mode)

    def sample(self, sample_shape=()):
        return symexp(super().rsample(sample_shape))

    def log_prob(self, x):
        return super().log_prob(symlog(x))


class DecoderDist:
    """
    Distribution for decoder output, which is a mixture of Gaussian for most
    of the states and a Categorical for the last four contact states which are
    parameterised by 8 logits.
    """

    def __init__(self, x):
        self.obs_dim = x.shape[-1]
        self.obs_dist = torchd.Normal(x[..., :-4], 1)
        self.contact_dist = torchd.Bernoulli(logits=x[..., -4:])

    @property
    def mode(self):
        obs = self.obs_dist.mode
        contacts = self.contact_dist.mode
        return torch.cat([obs, contacts], dim=-1)

    def log_prob(self, x):
        obs_lp = self.obs_dist.log_prob(x[..., :-4])
        contact_lp = self.contact_dist.log_prob(x[..., -4:])
        return torch.cat([obs_lp, contact_lp], dim=-1).sum(dim=-1)


class CategoricalDist:
    def __init__(self, logits, unimix_ratio, dim):
        probs = F.softmax(logits, dim=-1)
        probs = probs * (1 - unimix_ratio) + (unimix_ratio / probs.shape[-1])
        self.logits = torch.log(probs)
        self.shape = self.logits.shape[:-1]
        self.num_classes = dim

        self._dist = torchd.Independent(
            torchd.OneHotCategoricalStraightThrough(
                logits=logits.reshape(*self.shape, dim, dim)
            ),
            1,
        )

    def __gettattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self):
        return self._dist.rsample().reshape(*self.shape, self.num_classes**2)

    def log_prob(self, x):
        return self._dist.log_prob(
            x.reshape(*self.shape, self.num_classes, self.num_classes)
        )

    @property
    def dist(self):
        return self._dist


class TwoHotDistSymlog:
    def __init__(self, logits=None, low=-20.0, high=20.0, device="cuda"):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255

    def mode(self):
        _mode = self.probs * self.buckets
        return symexp(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element
    # in targets
    def log_prob(self, x):
        x = symlog(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)
