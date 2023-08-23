import sys
import time
import webbrowser
from datetime import datetime
from shutil import copy

import numpy as np
import torch
from ruamel.yaml import YAML
from tensorboard import program


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class FileSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir / datetime.now().strftime("%b-%d_%H-%M-%S") / "logging"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "../models").mkdir(parents=True, exist_ok=True)

        if save_items is not None:
            for save_item in save_items:
                copy(str(save_item), str(self._data_dir / ".."))

    @property
    def data_dir(self):
        return self._data_dir


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Timer:
    def __init__(self, real_delta, sleep):
        self._real_delta = real_delta
        self._sleep = sleep
        self._start = None

    def start(self):
        self._start = time.time()

    def end(self):
        delta = time.time() - self._start
        if (delta < self._real_delta) and self._sleep:
            time.sleep(self._real_delta - delta)


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


def tensorboard_launcher(directory_path):
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", directory_path])
    url = tb.launch()
    print("Tensorboard session created: " + url)
    webbrowser.open_new(url)


def init_config(config_path, args):
    with open(config_path, "r") as f:
        full_config_dict = YAML().load(f)
    config_dict = full_config_dict["default"]
    config_dict["env_name"] = args.env

    # raisim mode
    if config_dict["env_name"] == "raisim":
        for key, value in full_config_dict["raisim"].items():
            config_dict[key] = value

    # ditto
    config_dict["ditto"] = True
    for key, value in full_config_dict["ditto"].items():
        config_dict[key] = value

    # policy training mode
    if args.agent is not None:
        config_dict["log_every"] = 5e2
        config_dict["eval_every"] = 2e3
        config_dict["ditto_wm_steps"] = 0

    # debug mode
    if (
        hasattr(sys, "gettrace")
        and sys.gettrace() is not None
        and "debug" in full_config_dict
    ):
        print("debug mode")
        for key, value in full_config_dict["debug"].items():
            config_dict[key] = value

    config = Config(config_dict)
    config.time_limit //= config.action_repeat
    config.eval_steps //= config.action_repeat
    config.train_every = (config.batch_length * config.batch_size) // config.train_ratio

    return config, config_dict


def act_case(act):
    if act == "elu":
        activation = torch.nn.ELU
    elif act == "silu":
        activation = torch.nn.SiLU
    else:
        activation = torch.nn.ReLU
    return activation


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def load_expert_data(path, obs_dim, device):
    expert_data = torch.tensor(np.load(path)).to(torch.float32).to(device)
    obs = symlog(expert_data[..., :obs_dim])
    action = expert_data[..., obs_dim:]
    return {"obs": obs, "action": action}
