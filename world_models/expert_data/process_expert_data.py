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


data_dir = '10k'
obs_dim = 49
eval_eps = 50

# uncomment this if you're working with a bag file
# states = read_bag(data_dir, 'expert', ['state_and_action'])[0]
# states = states.to_numpy()[::16, 3:].astype(np.float32)

# load data
states = np.load(f'{data_dir}/expert_raw.npy')
states = np.delete(states, slice(obs_dim - 12, -12), axis=-1)

# split into episodes, the swapaxes are necessary to split correctly
states = states.swapaxes(0, 1)
states = states.reshape(-1, 200, obs_dim)
states = states.swapaxes(0, 1)
states = states[:-(states.shape[0] % 64)]  # must be divisible by 64
print(f'Expert data shape: {states.shape}')

# initialization data
init_data = np.zeros((states.shape[0], eval_eps, obs_dim))
init_data[..., 2] = states[:, -eval_eps:, 0]  # height
init_data[..., 3] = 1  # orientation
init_data[..., 7:19] = states[:, -eval_eps:, 4:16]  # joint angles
init_data[..., 19:37] = states[:, -eval_eps:, 16:34]  # joint velocities
init_data[..., -12:] = states[:, -eval_eps:, -12:]  # actions

# save two version of the eval data, one for initialization
# and one for encoding
np.save(data_dir + '/expert_init', init_data)
np.save(data_dir + '/expert_eval', states[:, -eval_eps:, 1:])

# remove validation eps and height
np.save(data_dir + '/expert', states[:, :-eval_eps, 1:])
