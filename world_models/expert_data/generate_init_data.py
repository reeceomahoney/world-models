from pathlib import Path
import numpy as np

home_path = Path(__file__).absolute().parents[2]
expert = 'fwd_step'
expert_data = np.load(str(home_path / 'world_models/expert_data' / expert / 'expert_w_height.npy'), allow_pickle=True)
init_data = np.zeros((expert_data.shape[0], 37))

# position
init_data[:, 2] = expert_data[:, 0]
init_data[:, 3] = 1
init_data[:, 7:19] = expert_data[:, 4:16]

# velocity
init_data[:, 19:] = expert_data[:, 16:34]
print(init_data[0])
np.savetxt(str(home_path / 'world_models/expert_data' / expert / 'init_data.csv'), init_data, delimiter=",")
