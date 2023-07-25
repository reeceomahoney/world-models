from pathlib import Path
import time
import socket

from torch.utils.tensorboard import SummaryWriter

from .utils import FileSaver


class Logger:
    """A class for logging data to tensorboard and printing to console.

    Methods
    -------
    log(tag, info)
        Adds the info to the info dict, prepending the tag to the key.
    publish(step)
        Publishes the info dict to tensorboard and prints the data listed in
        print_keys to console.
    """

    def __init__(self, config):
        self.config = config
        self.info = {}
        self.start_time = time.time()
        self.print_keys = ['world_model/pred_loss',
                           'world_model/kl_loss',
                           'actor_critic_loss/policy_loss',
                           'actor_critic_loss/value_loss',
                           'actor_reward/reward',
                           'eval/ditto_reward']

        log_dir = self._set_log_dir()
        self.writer = SummaryWriter(log_dir, flush_secs=10)

        # this avoids multilines being saved as separate runs
        layout = {'imag': {'reward_ema': ['Multiline', ['reward_ema/05',
                                                        'reward_ema/95']]},
                  'rewards': {'reward': ['Multiline', ['reward/mean',
                                                       'reward/var']]}}
        self.writer.add_custom_scalars(layout)

    def _set_log_dir(self):
        home_path = Path(__file__).parents[1].absolute()
        log_dir = home_path / 'logs' / self.config.env_name

        # server specific log directory
        if socket.gethostname() == 'bdemoss-3090':
            log_dir = Path('/data2/reece/raisim')

        # creates a new folder for each run and saves the config file
        saver = FileSaver(log_dir, [home_path / 'config.yaml'])
        return saver.data_dir

    def log(self, tag: str, info: dict):
        for k, v in info.items():
            self.info[f'{tag}/{k}'] = v

    def publish(self, step: int):
        dt = time.time() - self.start_time

        # log to tensorboard
        for k, v in self.info.items():
            self.writer.add_scalar(k, v, step)

        print("------------------------------------------")
        print(f"{'step: ':>21}{step}")
        print(f"{'time: ':>21}{dt:.3f}s")
        print("------------------------------------------")
        for k in self.print_keys:
            if k in self.info:
                print(f"{k:<30}{self.info[k]:>10.3f}")
        print("------------------------------------------\n")

        self.info = {}
