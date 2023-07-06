import torch

import world_models.common as common


def encode_and_store(agent, expert_sampler, eval=False):
    data = agent.encode_expert_data(expert_sampler, eval=eval)
    return common.LatentSampler(data)


def visualize_wm(agent, env_driver, latent_sampler):
    timer = common.Timer(0.04, sleep=True)
    env_driver.turn_on_visualization()
    for _ in range(3):
        env_driver.reset()
        data = latent_sampler.sample(1, 64)
        for i in range(data['state'].shape[0]):
            timer.start()
            obs_target = agent.world_model.decode(data['state'][i])
            env_driver.set_target(obs_target.detach().cpu().numpy())
            timer.end()
    env_driver.turn_off_visualization()


def main(config, env_driver, agent, expert_sampler, logger):
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.ditto_wm_eval_every)
    agent.set_actor_critic()

    print('\ntraining world model...')
    for step in range(int(config.ditto_wm_steps)):
        info = agent.train_world_model(expert_sampler)[-1]
        logger.log(info, step, should_log(step), False)

        if should_eval(step):
            latent_sampler = encode_and_store(agent, expert_sampler, eval=True)
            visualize_wm(agent, env_driver, latent_sampler)
            torch.save(agent.state_dict(),
                       f'{logger.writer.log_dir}/../models/wm_{step}.pt')

    print('\nimitation learning...')
    latent_sampler = encode_and_store(agent, expert_sampler)
    should_log = common.Every(config.log_every)
    should_eval = common.Every(config.eval_every)
    for step in range(int(config.ditto_il_steps)):
        info = agent.ditto_step(latent_sampler)
        logger.log(info, step, should_log(step), should_eval(step))
