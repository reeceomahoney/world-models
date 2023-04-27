import os
import argparse
import pickle

import numpy as np
import torch

import dreamer.common as common
from dreamer.agent import Agent

# paths
home_path = os.path.dirname(os.path.realpath(__file__)) + '/../'
dreamer_config_path = home_path + '/dreamer/config.yaml'
ditto_config_path = home_path + '/ditto/config.yaml'

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
args = parser.parse_args()

# config and env
config, config_dict = common.init_config(dreamer_config_path, args.env, ditto_config_path)
env_driver = common.get_driver(config, config_dict)
obs_dim = env_driver.env_info()[0]
print(f'using device: {config.device}')

agent = Agent(*env_driver.env_info(), config)
replay = common.ReplayBuffer(config, ('obs', 'cont', 'action'))
logger = common.Logger(config, agent, env_driver, replay)

# load expert data
expert_data = np.load(home_path + 'ditto/data/expert.npy')
step = 0
print('loading expert data...')
for i in range(expert_data.shape[0]):
    step += 1
    replay.store({'obs': expert_data[i, :obs_dim][np.newaxis],
                  'cont': np.array([1]),
                  'action': expert_data[i, obs_dim:][np.newaxis]})
    if step >= config.time_limit:
        replay.add_episode()
        step = 0

# world model training
print('\ntraining world model...')
should_log = common.Every(config.log_every)
for step in range(config.ditto_wm_steps):
    info = agent.train_world_model(replay)[-1]
    logger.log(info, step, should_log(step), False)
agent.encode_expert_data(replay)

# imitation learning
print('\nimitation learning...')
should_log = common.Every(config.log_every)
for step in range(config.ditto_il_steps):
    info = agent.ditto_step(replay)[-1]
    logger.log(info, step, should_log(step), False)
