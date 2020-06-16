import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime

moving_avg = False

global_running_r = 0
global_running_l = 0


# episodes_rewards = []


class Logger:
    simulation_steps = 0

    def __init__(self, **config):
        self.config = config
        self.moving_avg_window = 5
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.create_wights_folder()
        self.min_episode_reward = np.inf
        self.max_episode_reward = -np.inf
        self.avg_episode_reward = -np.inf
        self.avg_steps_reward = 0

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

    @staticmethod
    def create_wights_folder():
        if not os.path.exists("models"):
            os.mkdir("models")

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):

        episode, episode_reward, loss, steps, memory_length, epsilon = args
        # episodes_rewards.append(episode_reward)
        #
        # self.min_episode_reward = min(self.min_episode_reward, episode_reward)
        # self.max_episode_reward = max(self.max_episode_reward, episode_reward)
        # self.avg_episode_reward = sum(episodes_rewards) / self.episode
        #
        # self.avg_steps_reward = episode_reward / self.steps

        global global_running_r, global_running_l

        if global_running_r == 0:
            global_running_r = episode_reward
            global_running_l = loss
        else:
            # if moving_avg:
            #     if len(global_running_r) < self.moving_avg_window:
            #         global_running_r.append(0.99 * global_running_r[-1] + 0.01 * episode_reward)
            #         global_running_l.append(0.99 * global_running_l[-1] + 0.01 * loss)
            #     else:
            #         global_running_r.append(episode_reward)
            #         global_running_l.append(loss)
            #         weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
            #         *r, = np.convolve(global_running_r[1:], weights, 'valid')
            #         global_running_r[-1], *_ = r
            #         *l, = np.convolve(global_running_l[1:], weights, 'valid')
            #         global_running_l[-1], *_ = l
            # else:
            global_running_l = 0.99 * global_running_l + 0.01 * loss
            global_running_r = 0.99 * global_running_r + 0.01 * episode_reward

        memory = psutil.virtual_memory()

        if episode % self.config["interval"] == 0:
            print("EP:{}| "
                  "EP_Reward:{}| "
                  "EP_Running_Reward:{:3.3f}| "
                  "EP_Running_loss:{:3.3f}| "
                  "EP_Duration:{:3.3f}| "
                  "EP_loss:{:3.3f}| "
                  "Epsilon:{:.3f}| "
                  "Memory_length:{}| "
                  "Mean_steps_time:{:3.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time:{}".format(episode,
                                   episode_reward,
                                   global_running_r,
                                   global_running_l,
                                   self.duration,
                                   loss,  # TODO make loss smooth
                                   epsilon,
                                   memory_length,
                                   self.duration / steps,
                                   self.to_gb(memory.used),
                                   self.to_gb(memory.total),
                                   datetime.datetime.now().strftime("%H:%M:%S")
                                   ))
        with SummaryWriter("./logs/" + self.log_dir) as writer:
            writer.add_scalar("Loss", loss, self.simulation_steps)
            writer.add_scalar("Episode running reward", global_running_r, self.simulation_steps)
            # writer.add_hparams({
            #     "lr": 0.005},
            #     {"hparam/loss": loss})

        # print("Min episode reward:{:3d}| "
        #       "Max episode reward:{:3d}| "
        #       "Average episode reward:{:0.3f}| "
        #       "Average steps reward:{:0.3f}| "
        #       "Episode Duration:{:0.3f}| "
        #       "Episode loss:{:0.3f}| "
        #       "eps_threshold:{:0.3f}| "
        #       "step:{:3d}| "
        #       "mean steps time:{:0.3f}| "
        #       "{:.1f}/{:.1f} GB RAM".format(self.min_episode_reward,
        #                                     self.max_episode_reward,
        #                                     self.avg_episode_reward,
        #                                     self.avg_steps_reward,
        #                                     self.duration,
        #                                     loss,  # TODO make loss smooth
        #                                     eps_threshold,
        #                                     self.steps,  # it should be in each step not in each episode
        #                                     self.duration / self.steps,
        #                                     to_gb(memory.used),
        #                                     to_gb(memory.total)
        #                                     ))

    def save_weights(self, episode, agent):
        torch.save({"online_model_state_dict": agent.online_model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "memory": agent.memory,
                    "episode": episode},
                   self.config["weights_path"])

    def load_weights(self):
        checkpoint = torch.load(self.config["weights_path"])
        return checkpoint
