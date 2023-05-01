from .models import *
from .utils import *


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(WorldModel, self).__init__()
        layers = config.layers
        act = config.act
        device = config.device
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim

        # rssm core
        self._recurrent_model = RecurrentModel(self.z_dim + act_dim, self.h_dim, device)
        self._encoder = CategoricalMLP(self.h_dim + obs_dim, self.z_dim, config, device)
        self._dynamics = CategoricalMLP(self.h_dim, self.z_dim, config, device)

        # prediction
        self._decoder = Decoder(self.h_dim + self.z_dim, obs_dim, layers, act, device)
        self._reward_model = TwoHotSymlogMLP(config)
        self._cont_model = BernoulliMLP(self.h_dim + self.z_dim, 1, layers, act, device)

    def forward(self, h_t, z_t, action):
        # step recurrent state
        return self._recurrent_model(torch.cat((z_t, action), dim=-1), h_t)

    def dynamics(self, h_t):
        return self._dynamics(h_t).sample()

    def encode(self, h_t, obs):
        return self._encoder(torch.cat((h_t, obs), dim=-1)).sample()

    def decode(self, state):
        return self._decoder(state).mode

    def step(self, state, action):
        # step latent state in imagination
        h_t1 = self.forward(state[..., :self.h_dim], state[..., self.h_dim:], action)
        z_t1 = self.dynamics(h_t1)
        return torch.cat((h_t1, z_t1), dim=-1)

    def log_probs(self, data, states):
        log_probs = []
        if 'obs' in data:
            log_probs.append(self._decoder(states).log_prob(data['obs']))
        if 'reward' in data:
            log_probs.append(self._reward_model(states).log_prob(data['reward']))
        if 'cont' in data:
            log_probs.append(self._cont_model(states).log_prob(data['cont']))
        return torch.stack(log_probs)

    def get_z_dists(self, h_t, obs):
        return self._encoder(torch.cat((h_t, obs), dim=-1)), self._dynamics(h_t)

    def reward(self, state):
        return self._reward_model(state).mode()

    def cont(self, state):
        return self._cont_model(state).sample()

    def predict(self, state):
        obs_1 = symexp(self._decoder(state).mode)
        reward_1 = self._reward_model(state).mode()
        cont_1 = self._cont_model(state).sample()
        return obs_1, reward_1, cont_1
