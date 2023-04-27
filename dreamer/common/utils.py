import ntpath
import os
import sys
import webbrowser
from datetime import datetime
from shutil import copyfile

import torch
from ruamel.yaml import YAML
from tensorboard import program


class Config:
    def __init__(self, config_dict):

        for key, value in config_dict.items():
            setattr(self, key, value)


class FileSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.now().strftime('%b-%d_%H-%M-%S')
        os.makedirs(self._data_dir)
        os.makedirs(self._data_dir + '/models')

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

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


def tensorboard_launcher(directory_path):
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("Tensorboard session created: "+url)
    webbrowser.open_new(url)


def init_config(config_path, env_name, ditto_config_path=None):
    with open(config_path, 'r') as f:
        full_config_dict = YAML().load(f)
    config_dict = full_config_dict['default']
    config_dict['env_name'] = env_name

    # raisim mode
    if config_dict['env_name'] == 'raisim':
        for key, value in full_config_dict['raisim'].items():
            config_dict[key] = value

    # ditto mode
    if ditto_config_path:
        print('ditto mode')
        with open(ditto_config_path, 'r') as f:
            ditto_config_dict = YAML().load(f)['default']
        for key, value in ditto_config_dict.items():
            config_dict[key] = value

    # debug mode
    if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        print('debug mode')
        for key, value in full_config_dict['debug'].items():
            config_dict[key] = value

    config = Config(config_dict)
    config.time_limit //= config.action_repeat
    config.eval_steps //= config.action_repeat
    config.train_every = (config.batch_length * config.batch_size) // config.train_ratio

    return config, config_dict


def act_case(act):
    if act == 'elu':
        activation = torch.nn.ELU
    elif act == 'silu':
        activation = torch.nn.SiLU
    else:
        activation = torch.nn.ReLU
    return activation


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
