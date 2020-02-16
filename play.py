import torch
import numpy as np
from skimage.transform import resize
import gym
from gym import wrappers
import time
import psutil
import cv2


class Play:
    def __init__(self, agent, env, path):
        torch.cuda.empty_cache()
        # time.sleep(30)
        self.agent = agent
        self.path = path
        self.agent.ready_to_play(self.path)
        self.env = env
        self.env._max_episode_steps = 1000
        # self.env = wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.stacked_frames = np.zeros(shape=[84, 84, 4], dtype='float32')

        self.memory = psutil.virtual_memory()
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.VideoWriter = cv2.VideoWriter('output.avi', self.fourcc, 50.0, (160, 210))

    @staticmethod
    def rgb2gray(img):
        return 0.2125 * img[..., 0] + 0.7154 * img[..., 1] + 0.0721 * img[..., 2]

    def preprocessing(self, img):
        img = self.rgb2gray(img) / 255.0
        img = resize(img, output_shape=[84, 84])
        return img

    def stack_frames(self, stacked_frames, state, is_new_episode):
        frame = self.preprocessing(state)

        if is_new_episode:
            stacked_frames = np.stack([frame for _ in range(4)], axis=2)
        else:
            stacked_frames = stacked_frames[..., :3]
            stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=2)], axis=2)
        return stacked_frames

    def evaluate(self):

        print("--------Play mode--------")
        for _ in range(10):
            done = False
            state = self.env.reset()
            total_reward = 0
            self.stacked_frames = self.stack_frames(self.stacked_frames, state, True)
            i = 0

            while i < 1000 and not done:
                i += 1
                stacked_frames_copy = self.stacked_frames.copy()
                action = self.agent.get_action(stacked_frames_copy)
                next_state, r, done, info = self.env.step(action)
                self.stacked_frames = self.stack_frames(self.stacked_frames, next_state, False)
                # self.env.render()
                total_reward += r
                # time.sleep(0.05)
                # cv2.waitKey(20)
                self.VideoWriter.write(next_state)

            print("Total episode reward:", total_reward)
        # self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()
