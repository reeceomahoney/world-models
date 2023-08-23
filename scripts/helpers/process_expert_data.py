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


data_dir = 'fwd_cont_10k'
obs_dim = 53
eval_eps = 100

# uncomment this if you're working with a bag file
# states = read_bag(data_dir, 'expert', ['state_and_action'])[0]
# states = states.to_numpy()[::16, 3:].astype(np.float32)

# load csv
data_dir = Path(__file__).parent.absolute() / data_dir
states = pd.read_csv(str(data_dir / 'expert.csv'), header=None).to_numpy()
print(f'Expert data shape: {states.shape}')

# load data
# states = np.load(f'{data_dir}/expert_raw.npy')
# tmp = states[:5, 0]
# states = np.delete(states, slice(obs_dim - 12, -12), axis=-1)
# assert states[:5, 0, :obs_dim - 12].all() == tmp[:, obs_dim - 12].all()
# assert states[:5, 0, -12:].all() == tmp[:, -12:].all()

# split into episodes, the swapaxes is necessary to split correctly
tmp = states[:5]
tmp_2 = states[-5:]
states = states.reshape(-1, 200, obs_dim)
states = np.swapaxes(states, 0, 1)
assert (states[:5, 0] == tmp).all(), 'episodes are not split correctly'
assert (states[-5:, -1] == tmp_2).all(), 'episodes are not split correctly'

# must be divisible by 64 to work with encoder batching
states = states[:-(states.shape[0] % 64)]
assert states.shape[0] % 64 == 0
print(f'Expert data shape: {states.shape}')

# initialization data
init_data = np.zeros((states.shape[0], eval_eps, obs_dim - 13))
init_data[..., 2] = states[:, -eval_eps:, 0]  # height
init_data[..., 3] = 1  # orientation
init_data[..., 7:19] = states[:, -eval_eps:, 4:16]  # joint angles
init_data[..., 19:22] = states[:, -eval_eps:, 31:34]  # linear velocity
# angular + joint velocity
init_data[..., 22:37] = states[:, -eval_eps:, 16:31]
init_data[..., 37:40] = states[:, -eval_eps:, 34:37]  # vel cmd

# save two version of the eval data, one for initialization
# and one for encoding
np.save(data_dir / 'expert_init', init_data)
np.save(data_dir / 'expert_eval', states[:, -eval_eps:, 1:])

# remove validation eps and height
np.save(data_dir / 'expert', states[:, :-eval_eps, 1:])
