import torch

import world_models.common as common


def main(config, env_driver, agent, replays, logger):
    replay, state_replay = replays
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.ditto_wm_eval_every)
    agent.set_actor_critic()

    print('\ntraining world model...')
    for step in range(int(config.ditto_wm_steps)):
        info = agent.train_world_model(replay)[-1]
        logger.log(info, step, should_log(step), False)

        if should_eval(step):
            # encode and store expert data
            states = agent.encode_expert_data(replay)
            state_replay.store_all(states)

            timer = common.Timer(config.control_dt, sleep=True)
            env_driver.turn_on_visualization()
            for _ in range(3):
                env_driver.reset()
                data = state_replay.sample(1, 64)
                for i in range(data['state'].shape[0]):
                    timer.start()
                    obs_target = agent.world_model.decode(data['state'][i])
                    env_driver.set_target(obs_target.detach().cpu().numpy())
                    timer.end()
            env_driver.turn_off_visualization()
            torch.save(agent.state_dict(), f'{logger.writer.log_dir}/../models/agent_{step}.pt')

    # encode and store expert data
    states = agent.encode_expert_data(replay)
    states = {k: v.detach()for k, v in states.items()}
    state_replay.store_all(states)

    # imitation learning
    print('\nimitation learning...')
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.eval_every)
    for step in range(int(config.ditto_il_steps)):
        info = agent.ditto_step(state_replay)
        logger.log(info, step, should_log(step), should_eval(step))
