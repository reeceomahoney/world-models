import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

import dreamer


def setup():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="raisim")
    parser.add_argument("--agent", type=str, default=None)
    args = parser.parse_args()

    # paths
    home_path = Path(__file__).parents[1].absolute()
    config_path = home_path / "scripts/config.yaml"
    config, config_dict = dreamer.utils.init_config(config_path, args)
    expert_path = (
        home_path / "data" / config.ditto_dataset / "expert.npy"
    )
    print(f"Using expert data: {config.ditto_dataset}")

    # env
    env_driver = dreamer.driver.get_driver(config, config_dict)
    print(f"Using device: {config.device}")

    logger = dreamer.logger.Logger(config)

    # agent
    obs_dim, act_dim = env_driver.env_info()[:2]
    agent = dreamer.agent.Agent(*env_driver.env_info(), logger, config)

    if args.agent is not None:
        agent_state_dict = torch.load(
            home_path / args.agent, map_location=config.device
        )
        change_actor_critic = False
        if change_actor_critic:
            # These are for loading networks different to the current wm,
            # change bool if needed
            agent_state_dict = OrderedDict(
                [(k, v) for k, v in agent_state_dict.items() if "actor" not in k]
            )
            agent_state_dict = OrderedDict(
                [(k, v) for k, v in agent_state_dict.items() if "critic" not in k]
            )
        agent.load_state_dict(agent_state_dict, strict=False)

    # load training, evaluation, and initialisation data
    expert_data = dreamer.utils.load_expert_data(expert_path, obs_dim, config.device)
    expert_sampler = dreamer.storage.ExpertSampler(config, expert_data)

    expert_eval_path = expert_path.parent / "expert_eval.npy"
    expert_eval_data = dreamer.utils.load_expert_data(
        expert_eval_path, obs_dim, config.device
    )

    expert_init_path = expert_path.parent / "expert_init.npy"
    expert_init_data = (
        torch.tensor(np.load(expert_init_path)).to(torch.float32).to(config.device)
    )

    env_driver.load_expert_data(expert_init_data)
    agent.set_expert_data_size(expert_sampler)
    visualizer = dreamer.visualizer.Visualizer(
        config, agent, env_driver, logger, expert_eval_data
    )

    return config, agent, expert_sampler, logger, visualizer


def main():
    config, agent, expert_sampler, logger, visualizer = setup()
    # avoids loading exploration actor and critic
    agent.set_actor_critic()

    print("\nTraining world model...")
    for step in range(int(config.ditto_wm_steps)):
        agent.train_world_model(expert_sampler)

        if step % config.ditto_wm_eval_every == 0:
            visualizer.visualize_wm(step, expert_sampler)
        if step % config.log_every == 0:
            logger.publish(step)

    print("\nImitation learning...")
    for step in range(int(config.ditto_il_steps)):
        agent.ditto_step(expert_sampler)

        if step % config.eval_every == 0:
            visualizer.visualize_policy(step)
        if step % config.log_every == 0:
            agent.logger.publish(step)


if __name__ == "__main__":
    main()
