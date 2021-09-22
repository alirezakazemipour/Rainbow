import time
import numpy as np
import psutil
import torch
import os
import datetime
import glob
from collections import deque
from threading import Thread


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.experiment = self.config["experiment"]
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.running_loss = 0
        self.running_g_norm = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        if self.config["do_train"] and self.config["train_from_scratch"]:
            self.create_wights_folder(self.log_dir)
            self.experiment.log_parameters(self.config)
            # self.log_params()

        self.thread = Thread()

    @staticmethod
    def create_wights_folder(dir):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + dir)

    # def log_params(self):
    #     with SummaryWriter("Logs/" + self.log_dir) as writer:
    #         for k, v in self.config.items():
    #             writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):

        episode, episode_reward, loss, g_norm, step, beta, e_len = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_loss = loss
            self.running_g_norm = g_norm

        else:
            self.running_loss = 0.9 * self.running_loss + 0.1 * loss
            self.running_g_norm = 0.9 * self.running_g_norm + 0.1 * g_norm
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward

        self.last_10_ep_rewards.append(int(episode_reward))
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            last_10_ep_rewards = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')
        else:
            last_10_ep_rewards = 0  # It is not correct but does not matter.

        memory = psutil.virtual_memory()
        assert self.to_gb(memory.used) < 0.98 * self.to_gb(memory.total)

        if episode % (self.config["interval"] // 3) == 0:
            self.save_weights(episode, beta)

        if episode % self.config["interval"] == 0:
            print("E:{}| "
                  "E_Reward:{:.1f}| "
                  "E_Running_Reward:{:.2f}| "
                  "Running_loss:{:.2f}| "
                  "E_Duration:{:.2f}| "
                  "Mem_Len:{}| "
                  "Mean_steps_time:{:.2f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Beta:{:.2f}| "
                  "Time:{}| "
                  "Step:{}".format(episode,
                                   episode_reward,
                                   self.running_reward,
                                   self.running_loss,
                                   self.duration,
                                   len(self.agent.memory),
                                   self.duration / (step / episode),
                                   self.to_gb(memory.used),
                                   self.to_gb(memory.total),
                                   beta,
                                   datetime.datetime.now().strftime("%H:%M:%S"),
                                   step
                                   ))

        # with SummaryWriter("Logs/" + self.log_dir) as writer:
        metrics = {"Running episode reward": self.running_reward,
                   "Max episode reward": self.max_episode_reward,
                   "Moving last 10 episode rewards": last_10_ep_rewards,
                   "Running Grad Norm": self.running_g_norm,
                   "Running Loss": self.running_loss
                   }

        if self.thread.is_alive():
            self.thread.join()
        self.thread = Thread(target=self.log_metrics, args=(episode, metrics))
        self.thread.start()

    def log_metrics(self, episode, kwargs):
        for k, v in kwargs.items():
            self.experiment.log_metric(k, v, episode)

    def save_weights(self, episode, beta):
        torch.save({"online_model_state_dict": self.agent.online_model.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "episode": episode,
                    "beta": beta},
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        return checkpoint
