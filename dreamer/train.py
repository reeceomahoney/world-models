import os
import argparse
import pickle
from tqdm import tqdm

import torch

import common
from agent import Agent

# paths
home_path = os.path.dirname(os.path.realpath(__file__)) + '/../'
config_path = home_path + '/dreamer/config.yaml'

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--agent', type=str, default=None)
parser.add_argument('--replay', type=str, default=None)
args = parser.parse_args()

# config and env
config, config_dict = common.init_config(config_path, args.env)
env_driver = common.get_driver(config, config_dict)
print(f'using device: {config.device}')

# agent
agent = Agent(*env_driver.env_info(), config)
if args.agent is not None:
    agent_state_dict = torch.load(home_path + args.agent, map_location=config.device)
    agent.load_state_dict(agent_state_dict)

# replay buffer
if args.replay is None:
    replay = common.ReplayBuffer(config, ('obs', 'reward', 'cont', 'action'))
else:
    with open(home_path + args.replay, 'rb') as handle:
        replay = pickle.load(handle)
logger = common.Logger(config, agent, env_driver, replay)

# prefill buffer
if not config.zero_shot:
    print('prefilling buffer...')
    pbar = tqdm(total=config.prefill)
    obs, h_t, action = env_driver.reset()
    while len(replay) < config.prefill:
        obs, reward, done = env_driver(action)
        h_t, action = agent(h_t, obs)
        replay.store({'obs': obs, 'reward': reward, 'cont': 1 - done, 'action': action})
        if done.any() or env_driver.step >= config.time_limit:
            replay.add_episode()
            obs, h_t, action = env_driver.reset()
        # TODO: change this to only update when buffer fills up
        pbar.update(1)
    replay.add_episode()
    pbar.close()

    # pretrain
    print('\npretraining...')
    pbar = tqdm(total=config.pretrain)
    for step in range(config.pretrain):
        info = agent.train_step(step, replay, True)
        pbar.update(1)
    pbar.close()

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.eval_every)

    # train
    print('\ntraining...')
    obs, h_t, action = env_driver.reset()
    for step in range(int(config.steps)):
        obs, reward, done = env_driver(action)
        h_t, action = agent(h_t, obs)
        replay.store({'obs': obs, 'reward': reward, 'cont': 1 - done, 'action': action})

        info = agent.train_step(step, replay, should_train(step))
        logger.log(info, step, should_log(step), should_eval(step))

        if done.any() or env_driver.step >= config.time_limit:
            replay.add_episode()
            obs, h_t, action = env_driver.reset()

    for driver in [env_driver, logger.env_driver]:
        driver.close()

if config.zero_shot:
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.eval_every)

    print('zero-shot training...')
    for step in range(int(config.steps)):
        info = agent.train_step_zero_shot(replay)
        logger.log(info, step, should_log(step), should_eval(step))
