import random

from tqdm import tqdm

import world_models.common as common


def main(config, env_driver, agent, replay, logger):
    if len(replay) == 1:
        replay = replay[0]
    else:
        replay, expert_replay = replay

    if not config.zero_shot:
        print("prefilling buffer...")
        pbar = tqdm(total=config.prefill)
        obs, h_t, action = env_driver.reset()
        while len(replay) < config.prefill:
            obs, reward, done = env_driver(action)
            h_t, action = agent(h_t, obs)
            data = {"obs": obs, "cont": 1 - done, "action": action}
            if "reward" in replay.dims:
                data["reward"] = reward
            replay.store(data)

            if done.any() or env_driver.step >= config.time_limit:
                replay.add_episode()
                obs, h_t, action = env_driver.reset()
            pbar.n = len(replay)
            pbar.refresh()
        replay.add_episode()
        pbar.close()

        print("\npretraining...")
        pbar = tqdm(total=config.pretrain)
        for step in range(config.pretrain):
            if (
                "expert_replay" in locals()
                and random.random() < config.expert_replay_ratio
            ):
                sample_replay = expert_replay
            else:
                sample_replay = replay
            agent.train_step(step, sample_replay, True)
            pbar.update(1)
        pbar.close()

        should_train = common.Every(config.train_every)
        should_log = common.Every(config.log_every)
        should_eval = common.Every(config.eval_every)

        print("\ntraining...")
        obs, h_t, action = env_driver.reset()
        for step in range(int(config.steps)):
            obs, reward, done = env_driver(action)
            h_t, action = agent(h_t, obs)
            data = {"obs": obs, "cont": 1 - done, "action": action}
            if "reward" in replay.dims:
                data["reward"] = reward
            replay.store(data)

            if (
                "expert_replay" in locals()
                and random.random() < config.expert_replay_ratio
            ):
                sample_replay = expert_replay
            else:
                sample_replay = replay
            info = agent.train_step(step, sample_replay, should_train(step))
            info["buffer_size"] = len(replay)
            logger.log(info, step, should_log(step), should_eval(step))

            if done.any() or env_driver.step >= config.time_limit:
                replay.add_episode()
                obs, h_t, action = env_driver.reset()

        for driver in [env_driver, logger.env_driver]:
            driver.close()

    if config.zero_shot:
        should_log = common.Every(config.log_every)
        should_eval = common.Every(config.eval_every)

        print("zero-shot training...")
        for step in range(int(config.steps)):
            info = agent.train_step_zero_shot(replay)
            logger.log(info, step, should_log(step), should_eval(step))
