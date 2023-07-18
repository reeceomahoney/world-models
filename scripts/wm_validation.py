from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parents[1].absolute()))

from world_models import common
from world_models.agent import Agent

import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--wm', type=str, default=None)
args = parser.parse_args()
args.env = 'raisim'
args.ditto = True

home_path = Path(__file__).parents[1].absolute()
config_path = home_path / 'world_models/config.yaml'
config, config_dict = common.init_config(config_path, args)
config.ditto_wm_batch_size = 40

expert_path = home_path / 'world_models/expert_data' / config.ditto_dataset / \
    'expert_eval.npy'
expert_data = common.load_expert_data(expert_path, 36, config.device)
expert_sampler = common.ExpertSampler(config, expert_data)

agent = Agent(36, 12, 1, config)

wm_list = Path(args.wm).glob('wm*')
wm_list = sorted(wm_list, key=lambda x: int(x.stem.split('_')[-1]))

pred_loss_list, kl_loss_list = [], []
for wm in wm_list:
    print(f'loading {str(wm).split("/")[-1].split(".")[0]}')
    agent_state_dict = torch.load(wm, map_location=config.device)
    agent.load_state_dict(agent_state_dict, strict=False)

    pred_loss, kl_loss = 0, 0
    for _ in range(15):
        info = agent.train_world_model(expert_sampler, train=False)[-1]
        pred_loss += info['pred_loss'] / 15
        kl_loss += info['kl_loss'] / 15

    pred_loss_list.append(pred_loss)
    kl_loss_list.append(kl_loss)

# plot losses on two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(pred_loss_list)
ax1.set_title('Prediction Loss')
ax2.plot(kl_loss_list)
ax2.set_title('KL Loss')
plt.show()
