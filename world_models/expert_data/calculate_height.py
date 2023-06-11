import pinocchio as pin
from pathlib import Path
import numpy as np

home_path = Path(__file__).parents[2].absolute()
print(home_path / 'world_models/raisim_gym/rsc/anymal/urdf/anymal.urdf')
model = pin.buildModelFromUrdf(str(home_path / 'world_models/raisim_gym/rsc/anymal/urdf/anymal.urdf'))
data = model.createData()

expert = 'onphase'
expert_data = np.load(str(home_path / 'world_models/expert_data' / expert / 'expert.npy'), allow_pickle=True)
init_data = np.zeros((expert_data.shape[0], 37))

for i in range(expert_data.shape[0]):
    q = expert_data[i, 3:15]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    height = 0
    for j in range(model.njoints):
        height_tmp = -data.oMi[j].translation[2]
        if height_tmp > height:
            height = height_tmp

    # position
    init_data[i, 2] = height
    init_data[i, 3] = 1
    init_data[i, 7:19] = expert_data[i, 3:15]

    # velocity
    init_data[i, 19:] = expert_data[i, 15:33]

np.savetxt(str(home_path / 'world_models/expert_data' / expert / 'init_data.csv'), init_data, delimiter=",")
