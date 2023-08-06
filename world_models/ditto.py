import world_models.common as common


def encode_and_store(agent, expert_sampler, eval_ep=False):
    data = agent.encode_expert_data(expert_sampler, eval_ep=eval_ep)
    return common.LatentSampler(data)


def main(config, env_driver, agent, expert_sampler, logger, visualizer):
    agent.set_actor_critic()

    print('\nTraining world model...')
    for step in range(int(config.ditto_wm_steps)):
        agent.train_world_model(expert_sampler)[-1]

        if step % config.ditto_wm_eval_every == 0:
            latent_sampler = encode_and_store(agent, expert_sampler,
                                              eval_ep=True)
            visualizer.visualize_wm(step, latent_sampler)
        if step % config.log_every == 0:
            logger.publish(step)

    print('\nImitation learning...')
    latent_sampler = encode_and_store(agent, expert_sampler)
    for step in range(int(config.ditto_il_steps)):
        agent.ditto_step(latent_sampler)

        if step % config.eval_every == 0:
            visualizer.visualize_policy(step)
        if step % config.log_every == 0:
            agent.logger.publish(step)
