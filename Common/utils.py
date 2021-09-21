import numpy as np
import cv2
import gym


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(img, size=(84, 84)):
    img = rgb2gray(img)  # / 255.0 -> Do it later in order to open up more RAM !!!!
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def make_state(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames


def make_atari(env_id, seed):
    main_env = gym.make(env_id)
    assert 'NoFrameskip' in main_env.spec.id
    env = NoopResetEnv(main_env)
    env = MaxAndSkipEnv(env)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in main_env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    np.random.seed(seed)
    env.seed(seed)
    env.observation_space.np_random.seed(seed)
    env.action_space.np_random.seed(seed)

    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()

        noops = np.random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    # def get_rng_state(self):
    #     env_rng = self.env.np_random.get_state()
    #     env_obs_rng = self.env.observation_space.np_random.get_state()
    #     env_ac_rng = self.env.action_space.np_random.get_state()
    #     return np.random.get_state(), env_rng, env_obs_rng, env_ac_rng
    #
    # def set_rng_state(self, *state):
    #     np.random.set_state(state[0])
    #     self.env.np_random.set_state(state[1])
    #     self.env.observation_space.np_random.set_state(state[2])
    #     self.env.action_space.np_random.set_state(state[3])


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)

        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        reward = 0
        done = None
        for i in range(self.skip):
            obs, r, done, info = self.env.step(action)

            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            reward += r
            if done:
                break

        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done

        lives = info["ale.lives"]
        if self.lives > lives > 0:
            done = True

        self.lives = lives
        return obs, reward, done, info

    def reset(self):

        if self.real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs