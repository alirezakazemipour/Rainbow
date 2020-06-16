import numpy as np
import cv2


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(img):
    img = rgb2gray(img) / 255.0
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=2)
    else:
        stacked_frames = stacked_frames[..., :3]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=2)], axis=2)
    return stacked_frames


class AtariEnv:
    def __init__(self, env):
        self.noop_max = 30
        self.noop_action = 0
        self.env = env
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        state = None
        for _ in range(noops):
            state, _, done, _ = self.env.step(self.noop_action)
            if done:
                state = self.env.reset()

        return state

    def step(self, action):
        return self.env.step(action)


class RepeatActionEnv(AtariEnv):
    def __init__(self, env):
        super(RepeatActionEnv, self).__init__(env)
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self):
        return super().reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = super().step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info


class EpisodicLifeEnv(RepeatActionEnv):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env.gym_env)
        self.natural_done = True
        self.lives = self.env.ale.lives()

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.natural_done = done

        if self.lives > info["ale.lives"] > 0:
            done = True
        self.lives = info["ale.lives"]

        return state, reward, done, info

    def reset(self):
        if self.natural_done:
            state = super().reset()
        else:
            state, _, _, _ = super().step(0)
        self.lives = self.env.ale.lives()
        return state


class FireResetEnv(EpisodicLifeEnv):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.gym_env = env
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return super().step(action)

    def reset(self):
        super().reset()
        state, _, done, _ = super().step(1)
        if done:
            super().reset()
        state, _, done, _ = super().step(2)
        if done:
            super().reset()
        return state
