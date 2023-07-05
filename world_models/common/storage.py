import random

import numpy as np
import torch

size = lambda x: len(next(iter(x.values())))


class ReplayBuffer:
    def __init__(self, config, dims):
        self.current_episode = None
        self.episodes = []
        self.dims = dims
        self.num_envs = config.num_envs
        self.size = config.replay_size
        self.chunk_length = config.batch_length
        self.device = config.device

        self.clear_current_episode()

    def __len__(self):
        return sum([size(eps) for eps in self.episodes]) + size(self.current_episode) * self.num_envs

    # --------------------------------------------------------------------------------------------------------------
    # Storage

    def store(self, step):
        # step is a dict of 2D tensors
        self.current_episode = {k: torch.cat([self.current_episode[k], v.unsqueeze(0)], dim=0)
                                for k, v in step.items()}
        self._enforce_limit()

    def store_all(self, data):
        # self.episodes.append(data)
        self.episodes = [data]

    def store_all_from_tensors(self, data):
        data = {k: [x.squeeze(1) for x in torch.split(v, 1, dim=1)] for k, v in data.items()}
        self.episodes = [{k: v[i] for k, v in data.items()} for i in range(size(data))]

    def add_episode(self):
        if size(self.current_episode) >= self.chunk_length:
            for env in range(self.num_envs):
                self.episodes.append({k: v[:, env] for k, v in self.current_episode.items()})
        self.clear_current_episode()

    def clear_current_episode(self):
        self.current_episode = {k: torch.empty((0, self.num_envs, v)).to(self.device) for k, v in self.dims.items()}

    # --------------------------------------------------------------------------------------------------------------
    # Sampling

    def sample(self, batch, chunk_length=None):
        chunk_length = chunk_length or self.chunk_length
        all_eps = self._get_all_eps(chunk_length)

        eps = random.choices(all_eps, k=batch)
        start = np.random.randint([size(ep) - chunk_length + 1 for ep in eps], size=batch)
        end = start + chunk_length
        samples = {k: [ep[k][s:e] for ep, s, e in zip(eps, start, end)] for k in self.dims.keys()}
        return {k: torch.stack(v).swapaxes(0, 1) for k, v in samples.items()}

    def sample_all(self, chunk_length):
        samples = {k: torch.split(self.episodes[0][k], chunk_length, dim=0) for k in self.dims.keys()}
        return {k: torch.stack(v).swapaxes(0, 1) for k, v in samples.items()}

    def _enforce_limit(self):
        n = len(self) - self.size
        while n > 0:
            n -= size(self.episodes[0])
            self.episodes.pop(0)

    def _get_all_eps(self, chunk_length):
        # append current episode if its long enough
        all_eps = []
        if size(self.current_episode) >= chunk_length:
            for env in range(self.num_envs):
                all_eps.append({k: v[:, env] for k, v in self.current_episode.items()})
        all_eps += self.episodes
        return all_eps


class ExpertSampler:
    """
     Sequential sampler for the expert data.
    """
    def __init__(self, config, data):
        assert size(data) % config.batch_length == 0, \
            "Number of samples must be divisible by batch length"

        self.data = data
        self.batch_length = config.batch_length
        self.n_samples = size(data) // config.batch_length
        self.n_batches = data['obs'].shape[1]
        self.batch_size = config.ditto_wm_batch_size
        self.batch_idx = 0
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_samples:
            self.batch_idx = np.random.randint(
                self.n_batches - self.batch_size)
            self.idx = 0
        end_idx = self.batch_idx + self.batch_size
        samples = {k: v[self.idx*self.batch_length:
                        (self.idx+1)*self.batch_length, self.batch_idx:end_idx]
                   for k, v in self.data.items()}
        self.idx += 1
        return samples

    def get_slice(self, start, end):
        return {k: v[:, start:end] for k, v in self.data.items()}


class LatentSampler:
    """
    Random sampler for the encoded expert data.
    """
    def __init__(self, data):
        self.data = data
        self.len_samples, self.n_samples = data['state'].shape[:2]

    def __iter__(self):
        return self

    def sample(self, batch_size, batch_length):
        idx = torch.randint(self.n_samples, (batch_size,))
        start = torch.randint(self.len_samples - batch_length + 1,
                              (batch_size,))
        end = start + batch_length
        return {k: v[s:e, idx] for s, e in zip(start, end)
                for k, v in self.data.items()}
