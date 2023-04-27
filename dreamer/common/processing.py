import random

import torch

import dreamer.common as common


size = lambda x: len(next(iter(x.values())))


class ReplayBuffer:
    def __init__(self, config, data_terms):
        # TODO: might be a bug in multi env support
        # TODO: make this work better with different data terms
        self.steps = None
        self.episodes = []
        self._data_terms = data_terms

        self.num_envs = config.num_envs
        self.size = config.replay_size
        self.time_limit = config.time_limit
        self.chunk_length = config.batch_length
        self.episode_steps = 0
        self.device = config.device

        self.clear()

    def __len__(self):
        return sum([size(eps) for eps in self.episodes]) + size(self.steps) * self.num_envs

    def store(self, data):
        to_ten = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)

        step = {'obs': common.symlog(to_ten(data['obs'])) if 'obs' in self._data_terms else None,
                'reward': to_ten(data['reward']).unsqueeze(-1) if 'reward' in self._data_terms else None,
                'cont': to_ten(data['cont']).unsqueeze(-1) if 'cont' in self._data_terms else None,
                'action': to_ten(data['action']) if 'action' in self._data_terms else None}
        step = {k: v for k, v in step.items() if v is not None}

        for k in self._data_terms:
            self.steps[k].append(step[k])
        self._enforce_limit()

    def add_episode(self):
        # only add episode if we have enough steps
        if size(self.steps) > self.chunk_length:
            for env in range(self.num_envs):
                self.episodes.append({k: torch.stack(v)[:, env] for k, v in self.steps.items()})
        self.clear()

    def clear(self):
        self.steps = {k: [] for k in self._data_terms}

    def sample(self, batch, chunk_length=None):
        chunk_length = chunk_length or self.chunk_length
        if 'state' in self.episodes[0]:
            samples = {'state': []}
        else:
            samples = {k: [] for k in self.steps.keys()}
        all_eps = self._get_all_eps(chunk_length)

        for n in range(batch):
            eps = random.choices(all_eps)[0]
            start = random.randint(0, size(eps) - chunk_length)
            end = start + chunk_length
            for k in samples.keys():
                samples[k].append(eps[k][start:end])
        return {k: torch.stack(v).swapaxes(0, 1) for k, v in samples.items()}

    def sample_all(self, chunk_length):
        samples = {k: [] for k in self.steps.keys()}
        all_eps = self._get_all_eps(chunk_length)

        # split episodes into chunks
        for k in samples.keys():
            for eps in all_eps:
                samples[k] += list(torch.split(eps[k], chunk_length, dim=0))
            samples[k] = [x for x in samples[k] if x.shape[0] == chunk_length]  # remove small chunks
            samples[k] = torch.stack(samples[k]).swapaxes(0, 1)
        return samples

    def set_buffer(self, data):
        data = list(torch.split(next(iter(data.values())), 1, dim=1))
        data = [torch.squeeze(x, dim=1) for x in data]
        self.episodes = [{'state': v} for v in data]
        self.clear()

    def _enforce_limit(self):
        n = len(self) - self.size
        while n > 0:
            if self.episodes:
                n -= size(self.episodes[0])
                self.episodes.pop(0)
            else:
                for k in self.steps.keys():
                    del self.steps[k][:n]
                n = 0

    def _get_all_eps(self, chunk_length):
        # Append current steps to episodes if we have enough
        all_eps = []
        if size(self.steps) >= chunk_length:
            for env in range(self.num_envs):
                all_eps.append({k: torch.stack(v)[:, env] for k, v in self.steps.items()})
        all_eps += self.episodes
        return all_eps


class RewardEMA:
    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()
