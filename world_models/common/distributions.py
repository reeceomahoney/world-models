from abc import ABC

import torch.distributions as D
import torch.nn.functional as F

from .utils import *


class TruncatedNormal(D.Normal, ABC):
    def __init__(self, mean, std, low, high):
        super(TruncatedNormal, self).__init__(mean, std)
        self.low = low
        self.high = high

    def sample(self, sample_shape=()):
        return torch.clip(super().rsample(sample_shape), self.low, self.high)

    def log_prob(self, x):
        return super().log_prob(x)


class SymlogGaussian(D.Normal, ABC):
    def __init__(self, mean, std):
        super(SymlogGaussian, self).__init__(symlog(mean), std)

    def mode(self):
        return symexp(super().mode)

    def sample(self, sample_shape=()):
        return symexp(super().rsample(sample_shape))

    def log_prob(self, x):
        return super().log_prob(symlog(x))


class CategoricalDist:
    def __init__(self, logits, unimix_ratio, dim):
        probs = F.softmax(logits, dim=-1)
        probs = probs * (1 - unimix_ratio) + (unimix_ratio / probs.shape[-1])
        self.logits = torch.log(probs)

        logits = self.logits.reshape(*self.logits.shape[:-1], dim, dim)
        dist = D.OneHotCategoricalStraightThrough(logits=logits)
        dist = D.Independent(dist, 1)
        self._dist = dist
        self._dim = dim

    def sample(self):
        return self._dist.rsample().reshape(*self.logits.shape[:-1], self._dim ** 2)

    def entropy(self):
        return self._dist.entropy()

    @property
    def dist(self):
        return self._dist


class TwoHotDistSymlog:
    def __init__(self, logits=None, low=-20.0, high=20.0, device='cuda'):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255

    def mode(self):
        _mode = self.probs * self.buckets
        return symexp(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = symlog(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum((self.buckets > x[..., None]).to(torch.int32), dim=-1)
        below = torch.clip(below, 0, len(self.buckets)-1)
        above = torch.clip(above, 0, len(self.buckets)-1)
        equal = (below == above)

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
                F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None] +
                F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None])
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)
