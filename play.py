import torch
import numpy as np
from skimage.transform import resize


class Play:
    def __init__(self, agent, env, path):
        torch.cuda.empty_cache()
        self.agent = agent
        self.path = path
        self.agent.ready_to_play(self.path)
        self.env = env
        self.stacked_frames = np.zeros(shape=[84, 84, 4], dtype='float32')

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
