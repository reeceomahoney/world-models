import world_models.common as common


def main(config, env_driver, agent, replay, state_replay, logger):
    # world model training
    print('\ntraining world model...')
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.ditto_wm_eval_every)
    for step in range(int(config.ditto_wm_steps)):
        info = agent.train_world_model(replay)[-1]
        logger.log(info, step, should_log(step), False)

        if should_eval(step):
            # encode and store expert data
            states = agent.encode_expert_data(replay)[-1]
            state_replay.store_all_from_tensors(states)

            timer = common.Timer(config.control_dt, sleep=True)
            env_driver.turn_on_visualization()
            for _ in range(10):
                h_t = env_driver.reset()[1]
                data = state_replay.sample(1, config.imag_horizon + 1)
                for i in range(next(iter(data.values())).shape[0]):
                    timer.start()
                    obs_target = agent.world_model.decode(data['state'][i])
                    env_driver.set_target(obs_target.detach().cpu().numpy())
                    timer.end()
            env_driver.turn_off_visualization()

    # encode and store expert data
    states = agent.encode_expert_data(replay)[-1]
    state_replay.store_all_from_tensors(states)

    # imitation learning
    print('\nimitation learning...')
    config.log_every = 1e3
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.eval_every)
    for step in range(int(config.ditto_il_steps)):
        info = agent.ditto_step(state_replay)
        logger.log(info, step, should_log(step), should_eval(step))
