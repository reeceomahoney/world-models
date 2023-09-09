from .utils import Timer, symlog, symexp

import numpy as np
import gymnasium as gym
import torch
from PIL import Image


class Visualizer:
    """A class for visualizing the world model and policy.

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
        self.tot_steps = self.config.eval_eps * self.config.eval_steps

    def visualize_wm(self, step, expert_sampler):
        print("Visualizing world model...")
        timer = Timer(0.04, sleep=True)
        self.env_driver.turn_on_visualization()

        for _ in range(3):
            self.env_driver.reset()
            data = expert_sampler.sample(64, 1)
            data = self.agent.encode_data(data)[0]
            if self.config.wm_visualization == "prior":
                states_t = data["state"][0]

            for i in range(data["state"].shape[0]):
                timer.start()
                if self.config.wm_visualization == "post":
                    obs_target = symexp(self.agent.world_model.decode(data["state"][i]))
                elif self.config.wm_visualization == "prior":
                    states_t = self.agent.world_model.step(states_t, data["action"][i])
                    obs_target = symexp(self.agent.world_model.decode(states_t))
                else:
                    raise NotImplementedError
                self.env_driver.set_target(obs_target.detach().cpu().numpy())
                timer.end()

        self.env_driver.turn_off_visualization()
        torch.save(self.agent.state_dict(), f"{self.log_dir}/../models/wm_{step}.pt")

    def visualize_policy(self, step):
        print("Visualizing policy...")
        self.eval_info = {
            "obs_error": 0,
            "ditto_reward": torch.empty((0, 1)).to(self.config.device),
            "ditto_reward_std": 0,
        }
        agent_states = []
        mean_reward = 0

        self.env_driver.turn_on_visualization()
        timer = Timer(self.config.control_dt, self.config.real_time_eval)

        for _ in range(self.config.eval_eps):
            agent_states = self._calculate_ditto_reward(agent_states)
            obs, h_t, _ = self.env_driver.reset()

            for t in range(self.config.eval_steps):
                timer.start()
                preds, state, action = self.agent.predict(h_t, obs)
                obs, reward, done = self.env_driver(action)
                self.eval_info["obs_error"] += (
                    torch.norm(preds[0] - obs) / self.tot_steps
                )
                mean_reward += reward / self.tot_steps
                agent_states.append(state)
                h_t = state[..., : self.config.h_dim]
                timer.end()
                if done.any():
                    agent_states = self._calculate_ditto_reward(agent_states)
                    obs, h_t, _ = self.env_driver.reset()

        self.eval_info["ditto_reward_std"] = self.eval_info["ditto_reward"].std()
        self.eval_info["ditto_reward"] = self.eval_info["ditto_reward"].mean()

        torch.save(
            self.agent.state_dict(), self.log_dir + f"/../models/agent_{step}.pt"
        )
        self.env_driver.turn_off_visualization()

        self.logger.log("eval", {"env_reward": mean_reward, **self.eval_info})

    def _calculate_ditto_reward(self, agent_states):
        # skip this if agent states is empty
        if len(agent_states) == 0:
            return []

        # get corresponding expert states
        start_idx = self.env_driver.start_idx + 1  # +1 because of reset
        eps_idx = self.env_driver.eps_idx
        expert_data = {
            k: v[start_idx : start_idx + self.config.eval_steps, eps_idx]
            for k, v in self.expert_eval_data.items()
        }
        expert_states = self.agent.encode_data(expert_data)[0]["state"]
        agent_states = torch.stack(agent_states, dim=0)

        if self.config.ditto_state == "deter":
            expert_states = expert_states[..., : self.config.h_dim]
            agent_states = agent_states[..., : self.config.h_dim]

        # to deal with early termination
        expert_states = expert_states[: agent_states.shape[0]]

        reward = torch.sum(expert_states * agent_states, dim=-1)
        reward /= (
            torch.maximum(
                torch.norm(expert_states, dim=-1), torch.norm(agent_states, dim=-1)
            )
            ** 2
        )

        # rewards are appended to we can weight them by the number of steps
        self.eval_info["ditto_reward"] = torch.cat(
            (self.eval_info["ditto_reward"], reward)
        )

        # reset agent states
        return []


class GymVisualizer:
    def __init__(self, config, agent, env_driver, logger, expert_eval_data):
        self.config = config
        self.agent = agent
        self.env_driver = env_driver
        self.logger = logger
        self.expert_eval_data = expert_eval_data

        self.log_dir = str(self.logger.writer.log_dir)
        self.eval_info = None
        self.tot_steps = self.config.eval_eps * self.config.eval_steps

    def visualize_wm(self, step, expert_sampler):
        torch.save(self.agent.state_dict(), f"{self.log_dir}/../models/wm_{step}.pt")

    def visualize_policy(self, step):
        print("Visualizing policy...")
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        mean_reward = 0

        for _ in range(5):
            obs, info = env.reset()
            obs = torch.tensor(obs).to(torch.float32).to(self.config.device)
            obs = obs.unsqueeze(0)
            _, h_t, _ = self.env_driver.reset()

            for t in range(200):
                # get image
                img = env.render()
                img = Image.fromarray(img)
                img = img.resize((64, 64))
                img = torch.tensor(np.array(img)).to(torch.float32).to(self.config.device)
                img = img / 255.0 - 0.5
                img = img.unsqueeze(0)

                preds, state, action = self.agent.predict(h_t, img)
                action = action[0].cpu().numpy()
                obs, reward, done = env.step(action)[:3]
                obs = torch.tensor(obs).to(torch.float32).to(self.config.device)
                obs = obs.unsqueeze(0)
                h_t = state[..., : self.config.h_dim]
                mean_reward += reward / 5
                if done:
                    break

        env.close()
        self.logger.log("eval", {"env_reward": mean_reward})
        torch.save(
            self.agent.state_dict(), self.log_dir + f"/../models/agent_{step}.pt"
        )
