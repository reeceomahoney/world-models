import os
import time
import pickle

from torch.utils.tensorboard import SummaryWriter

from .driver import *
from .utils import *


class Logger:
    def __init__(self, config, agent, env_driver, replay, launch_tb=False):
        # launch tensorboard
        home_path = os.path.dirname(os.path.realpath(__file__)) + '/../../'
        saver = FileSaver(log_dir=home_path + "logs/" + config.env_name,
                          save_items=[home_path + "dreamer/config.yaml"])
        tensorboard_launcher(saver.data_dir) if launch_tb else None

        self.writer = SummaryWriter(log_dir=saver.data_dir, flush_secs=10)
        os.makedirs(saver.data_dir + '/datasets')

        self.config = config
        self.agent = agent
        self.env_driver = env_driver
        self._replay = replay
        self.eval_info = None
        self.reward_info, self.reward_mean, self.reward_std = None, None, None
        self.start_time = time.time()

        self._reset() if config.env_name == 'raisim' else None

    def store_reward(self):
        envs_reward_info = self.env_driver.get_reward_info()

        for reward_info in envs_reward_info:
            for k in self.reward_info.keys():
                self.reward_info[k].append(reward_info[k])

    def log(self, info, step, log_step, eval_step):
        if self.config.env_name == 'raisim':
            self.store_reward()
        if eval_step:
            self._eval(step)
        if log_step:
            dt = time.time() - self.start_time
            self.start_time = time.time()
            self._write(info, step, eval_step)
            self._print(info, step, dt)
            self._reset() if self.config.env_name == 'raisim' else None
        if self.config.Plan2Explore and step >= self.config.explore_steps:
            self.config.Plan2Explore = False

    def _reset(self):
        self.reward_info = {k: [] for k in self.config.reward.keys()}
        self.reward_mean, self.reward_std = {}, {}

    def _eval(self, step):
        print('evaluating...')
        self.eval_info = {'obs_error': 0, 'reward_error': 0, 'gamma_error': 0, 'eval_reward': 0}
        if self.config.env_name == 'raisim':
            eval_driver = self.env_driver
            eval_driver.turn_on_visualization()
            self._replay.add_episode()
        else:
            eval_driver = GymDriver(self.config, render=not self.config.ssh)

        for _ in range(self.config.eval_eps):
            obs, h_t, _ = eval_driver.reset()
            for t in range(self.config.eval_steps):
                start = time.time()
                preds, h_t, action = self.agent.predict(h_t, obs)
                obs, reward, done = eval_driver(action)
                self._update_eval_info(obs, reward, done, preds)
                delta = time.time() - start
                sim_delta = 0.04 * self.config.action_repeat
                if delta < sim_delta and self.config.real_time_eval:
                    time.sleep(sim_delta - delta)
                if done.any():
                    obs, h_t, _ = eval_driver.reset()

        torch.save(self.agent.state_dict(), self.writer.log_dir + f'/models/agent_{step}.pt')
        with open(self.writer.log_dir + f'/datasets/replay_{step}.pickle', 'wb') as f:
            pickle.dump(self._replay, f, protocol=pickle.HIGHEST_PROTOCOL)
        eval_driver.turn_off_visualization() if self.config.env_name == 'raisim' else eval_driver.close()

    def _update_eval_info(self, obs, reward, done, preds):
        preds = [p.detach().cpu().numpy() for p in preds]
        n = self.config.eval_eps * self.config.eval_steps
        self.eval_info['obs_error'] += np.mean(np.square(preds[0] - obs)) / n
        self.eval_info['reward_error'] += np.mean(np.square(preds[1] - reward)) / n
        self.eval_info['gamma_error'] += np.mean(np.square(preds[2] - (1 - done))) / n
        self.eval_info['eval_reward'] += reward.mean() / n

    def _write(self, info, step, eval_step):
        # losses
        for name in ['pred_loss', 'kl_loss', 'policy_loss', 'value_loss']:
            self._write_scalar(name, 'loss', info, step)
        if self.config.Plan2Explore:
            self.writer.add_scalar('loss/ensemble_loss', info['ensemble_loss'], step)

        # evaluation
        if eval_step:
            for name in ['obs_error', 'reward_error', 'gamma_error']:
                self._write_scalar(name, 'evaluation', self.eval_info, step)

        # imagination
        for name in ['imag_reward', 'imag_values', 'imag_value_targets']:
            self._write_scalar(name, 'imag', info, step)
        if 'reward_ema' in info:
            self.writer.add_scalars('imag/reward_ema', info['reward_ema'], step)

        if self.config.env_name == 'raisim':
            for k in self.reward_info.keys():
                self.reward_mean[k] = np.mean(np.array(self.reward_info[k]))
                self.reward_std[k] = np.std(np.array(self.reward_info[k]))
            self.writer.add_scalars('rewards/mean', self.reward_mean, step)
            self.writer.add_scalars('rewards/std', self.reward_std, step)

        # misc
        for name in ['act_std', 'buffer_size', 'act_size', 'entropy']:
            self._write_scalar(name, 'misc', info, step)

    def _write_scalar(self, name, section, info, step):
        if name in info:
            self.writer.add_scalar(f'{section}/{name}', info[name], step)

    def _print(self, info, step, dt):
        print("-----------------------------------------")
        print(f"step: {step}")
        print(f"time: {dt:.3f}s")
        print("-----------------------------------------")
        print(f"world model loss: {info['pred_loss']:.3f}  kl loss: {info['kl_loss']:.3f}")
        if 'policy_loss' and 'value_loss' in info:
            print(f"policy loss: {info['policy_loss'].item():.3f}  value loss: {info['value_loss'].item():.3f}")

        if self.config.Plan2Explore:
            print(f"ensemble loss: {info['ensemble_loss'].item():.3f}")

        if 'imag_reward' in info:
            print(f"img tot reward: {info['imag_reward'].item():.3f}")
        print("-----------------------------------------\n")
