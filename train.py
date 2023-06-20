import argparse
import pickle
from pathlib import Path
from collections import OrderedDict

import torch

import world_models.dreamer as dreamer
import world_models.ditto as ditto

from world_models.agent import Agent
from world_models import common


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--ditto', type=str, default=False)
parser.add_argument('--agent', type=str, default=None)
parser.add_argument('--replay', type=str, default=None)
args = parser.parse_args()

# paths
home_path = Path(__file__).parent.absolute()
config_path = home_path / 'world_models/config.yaml'
config, config_dict = common.init_config(config_path, args)
expert_path = home_path / 'world_models/expert_data' / config.ditto_dataset / 'expert.npy'
print(f'using expert data: {config.ditto_dataset}')

# env
env_driver = common.get_driver(config, config_dict)
print(f'using device: {config.device}')

# agent
obs_dim, act_dim = env_driver.env_info()[:2]
agent = Agent(*env_driver.env_info(), config)
if args.agent is not None:
    agent_state_dict = torch.load(home_path / args.agent, map_location=config.device)
    # These are for loading wms different to the current model, uncomment if needed
    # agent_state_dict = OrderedDict([(k, v) for k, v in agent_state_dict.items() if not k.startswith('actor')])
    # agent_state_dict = OrderedDict([(k, v) for k, v in agent_state_dict.items() if 'actor' not in k])
    # agent_state_dict = OrderedDict([(k, v) for k, v in agent_state_dict.items() if 'critic' not in k])
    agent.load_state_dict(agent_state_dict, strict=False)

# replay buffer
if args.replay is None:
    if args.ditto:
        expert_data = common.load_expert_data(expert_path, obs_dim, config.device)
        replay = common.ExpertSampler(config, expert_data)
        state_replay = common.ReplayBuffer(config, {'state': config.h_dim + config.z_dim,
                                                    'post': config.z_dim,
                                                    'action': act_dim})
        replays = (replay, state_replay)
    else:
        replays = tuple([common.ReplayBuffer(config, {'obs': obs_dim, 'reward': 1, 'cont': 1, 'action': act_dim})])
else:
    with open(home_path / args.replay, 'rb') as handle:
        replays = tuple([pickle.load(handle)])

# logger
logger = common.Logger(config, agent, env_driver, replays[0])

# train
if not args.ditto:
    dreamer.main(config, env_driver, agent, replays, logger)
else:
    ditto.main(config, env_driver, agent, replays, logger)
