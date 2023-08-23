def main(config, env_driver, agent, expert_sampler, logger, visualizer):
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
