import torch
import numpy as np
from skimage.transform import resize
from utils import *


class Play:
    def __init__(self, agent, env, path):
        torch.cuda.empty_cache()
        self.agent = agent
        self.path = path
        self.agent.ready_to_play(self.path)
        self.env = env
        self.stacked_frames = np.zeros(shape=[84, 84, 4], dtype='float32')

    def evaluate(self):

        print("--------Play mode--------")
        for _ in range(5):
            done = 0
            state = self.env.reset()
            total_reward = 0
            self.stacked_frames = self.stack_frames(self.stacked_frames, state, True)

            while not done:
                stacked_frames_copy = self.stacked_frames.copy()
                action = self.agent.get_action(stacked_frames_copy)
                next_state, r, done, _ = self.env.step(action)
                self.stacked_frames = self.stack_frames(self.stacked_frames, next_state, False)
                self.env.render()
                total_reward += r

            print("Total episode reward:", total_reward)
        self.env.close()
