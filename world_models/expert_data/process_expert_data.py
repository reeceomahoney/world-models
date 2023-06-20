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


data_dir = 'fwd_step_05_50'
# states = read_bag(data_dir, 'expert', ['state_and_action'])[0].to_numpy()[::16, 3:].astype(np.float32)
states = pd.read_csv(f'{data_dir}/expert.csv').to_numpy()[::8].astype(np.float32)
states = states[:-(states.shape[0] % 64)]  # must be divisible by 64

# init data
init_data = np.zeros((states.shape[0], 49))
init_data[:, 2] = states[:, 0]
init_data[:, 3] = 1
init_data[:, 7:19] = states[:, 4:16]

# velocity
init_data[:, 19:37] = states[:, 16:34]

# action
init_data[:, 37:] = states[:, 37:]

np.savetxt(data_dir + '/init_data.csv', init_data, delimiter=",")

# without height
states = states[:, 1:]
np.save(data_dir + '/expert', states)
