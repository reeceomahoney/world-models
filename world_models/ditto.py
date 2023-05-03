import argparse
import os

import common as common
from agent import Agent

# paths
home_path = os.path.dirname(os.path.realpath(__file__))

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
args = parser.parse_args()

# config and env
config, config_dict = common.init_config(home_path + '/config.yaml', args.env)
env_driver = common.get_driver(config, config_dict)
obs_dim, act_dim = env_driver.env_info()[:2]
print(f'using device: {config.device}')

agent = Agent(*env_driver.env_info(), config)
replay = common.ReplayBuffer(config, {'obs': obs_dim, 'action': act_dim})
state_replay = common.ReplayBuffer(config, {'state': config.h_dim + config.z_dim})
logger = common.Logger(config, agent, env_driver, replay)

expert_data = common.load_expert_data(home_path + '/ditto/data/expert.npy', obs_dim, config.device)
replay.store_all(expert_data)

# world model training
print('\ntraining world model...')
should_log = common.Every(config.log_every)
should_eval = common.Every(config.ditto_wm_eval_every)
for step in range(int(config.ditto_wm_steps)):
    info = agent.train_world_model(replay)[-1]
    logger.log(info, step, should_log(step), False)

    if should_eval(step):
        # encode and store expert data
        states = agent.encode_expert_data(replay)[-1]
        state_replay.store_all_from_tensors(states)

        timer = common.Timer(config.control_dt, sleep=True)
        env_driver.turn_on_visualization()
        for _ in range(10):
            h_t = env_driver.reset()[1]
            data = state_replay.sample(1, config.imag_horizon + 1)
            for i in range(next(iter(data.values())).shape[0]):
                timer.start()
                obs_target = agent.world_model.decode(data['state'][i])
                env_driver.set_target(obs_target.detach().cpu().numpy())
                timer.end()
        env_driver.turn_off_visualization()

# imitation learning
print('\nimitation learning...')
config.log_every = 1e3
should_log = common.Every(config.log_every)
should_eval = common.Every(config.eval_every)
for step in range(int(config.ditto_il_steps)):
    info = agent.ditto_step(replay)
    logger.log(info, step, should_log(step), should_eval(step))
