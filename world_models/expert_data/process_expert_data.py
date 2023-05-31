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


data_dir = 'onphase'
states = read_bag(data_dir, 'expert', ['state_and_action'])[0].to_numpy()[::16, 3:].astype(np.float32)
states = states[:-(states.shape[0] % 50)]  # must be divisible by 50
np.save(data_dir + '/expert', states)
