from pathlib import Path

import numpy as np
import pandas as pd
from bagpy import bagreader


def read_bag(bag_dir, bag_name, topics):
    b = bagreader(f'{bag_dir}/{bag_name}.bag')
    bag_info = []

    for t in topics:
        bag_info.append((Path(f'{bag_dir}/{bag_name}/{t.replace("/", "-")}.csv'), f'/{t}'))

    csvs = []
    for path, topic in bag_info:
        if path.is_file():
            csvs.append(pd.read_csv(path))
        else:
            print(f'Extracting {topic} from bag file...')
            msg = b.message_by_topic(topic)
            csvs.append(pd.read_csv(msg))

    return csvs


data_dir = 'rand_eps'
obs_dim = 49
# states = read_bag(data_dir, 'expert', ['state_and_action'])[0].to_numpy()[::16, 3:].astype(np.float32)
states = pd.read_csv(f'{data_dir}/expert.csv').to_numpy()[::16].astype(np.float32)
states = states.reshape(-1, 1000, obs_dim).swapaxes(0, 1)
states = states[:-(states.shape[0] % 64)]  # must be divisible by 64

# initialization data
init_data = np.zeros((states.shape[0], obs_dim))
init_data[:, 2] = states[:, -1, 0]  # height
init_data[:, 3] = 1  # orientation
init_data[:, 7:19] = states[:, -1, 4:16]  # joint angles
init_data[:, 19:37] = states[:, -1, 16:34]  # joint velocities
init_data[:, 37:49] = states[:, -1, 37:49]  # actions

np.savetxt(data_dir + '/init_data.csv', init_data, delimiter=",")
np.save(data_dir + '/expert_eval', np.expand_dims(states[:, -1, 1:], axis=1))  # eval eps

np.save(data_dir + '/expert', states[:, :-1, 1:])  # remove validation eps and height
