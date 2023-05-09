import ast
from pathlib import Path

import numpy as np
import pandas as pd
from bagpy import bagreader
from scipy.spatial.transform import Rotation as R


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


states, vel_cmd, joint_target = read_bag('pos_error_cmds', 'expert',
                           ['state_estimator/anymal_state', 'joy_manager/twist', 'joint_target'])
states = states[::16].reset_index(drop=True)  # sample at 25Hz
joint_target = joint_target[::16].reset_index(drop=True)  # sample at 25Hz
df = pd.concat([states, vel_cmd, joint_target], axis=1)

# make all arrays the same length
length = min(len(states), len(joint_target), len(vel_cmd))
df = df[:length - 106]  # robot fell over at the end of the bag file

obs_dim = 48
obs = np.zeros((len(df), obs_dim))

for idx, row in df.iterrows():
    quat = [row['pose.pose.orientation.' + i] for i in ['x', 'y', 'z', 'w']]
    rot_mat = R.from_quat(quat).as_matrix()
    obs[idx, :3] = rot_mat[2].T.tolist()  # z-axis of rotation matrix

    obs[idx, 3:15] = ast.literal_eval(row['joints.position'])  # joint positions

    lin_vel = np.array([row['twist.twist.linear.' + i] for i in ['x', 'y', 'z']])
    obs[idx, 15:18] = np.dot(rot_mat.T, lin_vel)  # linear velocity

    ang_vel = np.array([row['twist.twist.angular.' + i] for i in ['x', 'y', 'z']])
    obs[idx, 18:21] = np.dot(rot_mat.T, ang_vel)  # angular velocity

    obs[idx, 21:33] = ast.literal_eval(row['joints.velocity'])  # joint velocities
    obs[idx, 33:36] = [row['twist.linear.x'], row['twist.linear.y'], row['twist.angular.z']]  # velocity command

    obs[idx, 36:] = np.array([row['data_' + str(i)] for i in range(12)])  # joint targets

# save to file
np.save('pos_error_cmds/expert', obs)
