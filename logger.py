import time
import numpy as np
import psutil

moving_avg = False

global_running_r = []
global_running_l = []


class LOG:
    def __init__(self):
        self.moving_avg_window = 5

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def printer(self, *args, **kwargs):

        episode, episode_reward, loss, eps_threshold, steps = args

        if len(global_running_r) == 0:
            global_running_r.append(episode_reward)
            global_running_l.append(loss)
        else:
            if moving_avg:
                if len(global_running_r) < self.moving_avg_window:
                    global_running_r.append(0.99 * global_running_r[-1] + 0.01 * episode_reward)
                    global_running_l.append(0.99 * global_running_l[-1] + 0.01 * loss)
                else:
                    global_running_r.append(episode_reward)
                    global_running_l.append(loss)
                    weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
                    *r, = np.convolve(global_running_r[1:], weights, 'valid')
                    global_running_r[-1], *_ = r
                    *l, = np.convolve(global_running_l[1:], weights, 'valid')
                    global_running_l[-1], *_ = l
            else:
                global_running_l.append(0.99 * global_running_l[-1] + 0.01 * loss)
                global_running_r.append(0.99 * global_running_r[-1] + 0.01 * episode_reward)

        memory = psutil.virtual_memory()
        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

        print("Episode:{:3d}| "
              "Episode_Reward:{:3d}| "
              "Episode_Running_r:{:0.3f}| "
              "Episode_Running_l:{:0.3f}| "
              "Episode Duration:{:0.3f}| "
              "Episode loss:{:0.3f}| "
              "eps_threshold:{:0.3f}| "
              "step:{:3d}| "
              "mean steps time:{:0.3f}| "
              "{:.1f}/{:.1f} GB RAM".format(episode,
                                            episode_reward,
                                            global_running_r[-1],
                                            global_running_l[-1],
                                            self.duration,
                                            loss,  # TODO make loss smooth
                                            eps_threshold,
                                            steps,  # it should be in each step not in each episode
                                            self.duration / steps,
                                            to_gb(memory.used),
                                            to_gb(memory.total)
                                            ))
