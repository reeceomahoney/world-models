from .utils import Timer

import torch


class Visualizer:
    """ A class for visualizing the world model and policy.

    Methods
    -------
    visualize_wm(step, latent_sampler)
        Visualizes the decoded observations from the world model.
    visualize_policy(step)
        Visualizes the policy in the environment.
    """
    def __init__(self, config, agent, env_driver, logger, expert_eval_data):
        self.config = config
        self.agent = agent
        self.env_driver = env_driver
        self.logger = logger
        self.expert_eval_data = expert_eval_data

        self.log_dir = str(self.logger.writer.log_dir)
        self.eval_info = None

    def visualize_wm(self, step, latent_sampler):
        timer = Timer(0.04, sleep=True)
        self.env_driver.turn_on_visualization()

        for _ in range(3):
            self.env_driver.reset()
            data = latent_sampler.sample(1, 64)
            for i in range(data['state'].shape[0]):
                timer.start()
                obs_target = self.agent.world_model.decode(data['state'][i])
                self.env_driver.set_target(obs_target.detach().cpu().numpy())
                timer.end()

        self.env_driver.turn_off_visualization()
        torch.save(self.agent.state_dict(),
                   f'{self.log_dir}/../models/wm_{step}.pt')

    def visualize_policy(self, step):
        print('evaluating...')
        self.eval_info = {
            'obs_error': 0,
            'ditto_reward': torch.empty((0, 1)).to(self.config.device),
            'ditto_reward_std': 0}
        agent_states = []
        mean_reward = 0

        self.env_driver.turn_on_visualization()
        timer = Timer(self.config.control_dt, self.config.real_time_eval)
        n = self.config.eval_eps * self.config.eval_steps

        for _ in range(self.config.eval_eps):
            agent_states = self._calculate_ditto_reward(agent_states)
            obs, h_t, _ = self.env_driver.reset()

            for t in range(self.config.eval_steps):
                timer.start()
                preds, h_t, action = self.agent.predict(h_t, obs)
                obs, reward, done = self.env_driver(action)
                self.eval_info['obs_error'] += torch.norm(preds[0] - obs) / n
                mean_reward += reward / n
                agent_states.append(h_t)
                timer.end()
                if done.any():
                    agent_states = self._calculate_ditto_reward(agent_states)
                    obs, h_t, _ = self.env_driver.reset()

        self.eval_info['ditto_reward_std'] = \
            self.eval_info['ditto_reward'].std()
        self.eval_info['ditto_reward'] = \
            self.eval_info['ditto_reward'].mean()

        torch.save(self.agent.state_dict(),
                   self.log_dir + f'/../models/agent_{step}.pt')
        self.env_driver.turn_off_visualization()

        self.logger.log('eval', {'env_reward': mean_reward, **self.eval_info})

    def _calculate_ditto_reward(self, agent_states):
        # skip this if agent states is empty
        if len(agent_states) == 0:
            return []

        # get corresponding expert states
        start_idx = self.env_driver.start_idx + 1  # +1 because of reset
        eps_idx = self.env_driver.eps_idx
        expert_data = {
            k: v[start_idx:start_idx + self.config.eval_steps, eps_idx]
            for k, v in self.expert_eval_data.items()}
        expert_states = self.agent.encode_expert_data(
            expert_data)['state'][..., :self.config.h_dim]

        agent_states = torch.stack(agent_states, dim=0)

        # to deal with early termination
        expert_states = expert_states[:agent_states.shape[0]]

        reward = torch.sum(expert_states * agent_states, dim=-1)
        reward /= (torch.maximum(torch.norm(expert_states, dim=-1),
                                 torch.norm(agent_states, dim=-1)) ** 2)

        # rewards are appended to we can weight them by the number of steps
        self.eval_info['ditto_reward'] = torch.cat((
            self.eval_info['ditto_reward'], reward))

        # reset agent states
        return []
