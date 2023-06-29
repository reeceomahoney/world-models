from pathlib import Path

import numpy as np
import pandas as pd
from bagpy import bagreader


def read_bag(bag_dir, bag_name, topics):
    b = bagreader(f'{bag_dir}/{bag_name}.bag')
    bag_info = []

    for t in topics:
        bag_info.append((Path(
            f'{bag_dir}/{bag_name}/{t.replace("/", "-")}.csv'), f'/{t}'))

    csvs = []
    for path, topic in bag_info:
        if path.is_file():
            csvs.append(pd.read_csv(path))
        else:
            print(f'Extracting {topic} from bag file...')
            msg = b.message_by_topic(topic)
            csvs.append(pd.read_csv(msg))

    return csvs


data_dir = 'rand_start_eps_2'
obs_dim = 49
eval_eps = 50

# uncomment this if you're working with a bag file
# states = read_bag(data_dir, 'expert', ['state_and_action'])[0]
# states = states.to_numpy()[::16, 3:].astype(np.float32)

states = pd.read_csv(
    f'{data_dir}/expert.csv').to_numpy()[::16].astype(np.float32)
states = states.reshape(-1, 200, obs_dim).swapaxes(0, 1)
states = states[:-(states.shape[0] % 64)]  # must be divisible by 64
print(f'Expert data shape: {states.shape}')

# initialization data
init_data = np.zeros((states.shape[0], eval_eps, obs_dim))
init_data[..., 2] = states[:, -eval_eps:, 0]  # height
init_data[..., 3] = 1  # orientation
init_data[..., 7:19] = states[:, -eval_eps:, 4:16]  # joint angles
init_data[..., 19:37] = states[:, -eval_eps:, 16:34]  # joint velocities
init_data[..., 37:49] = states[:, -eval_eps:, 37:49]  # actions

# save data
init_data = init_data.reshape(-1, obs_dim)
np.savetxt(data_dir + '/init_data.csv', init_data, delimiter=",")

expert_eval_data = np.expand_dims(
    states[:, -eval_eps:, 1:].reshape(-1, obs_dim - 1), axis=1)
np.save(data_dir + '/expert_eval', expert_eval_data)  # eval eps

# remove validation eps and height
np.save(data_dir + '/expert', states[:, :-eval_eps, 1:])
