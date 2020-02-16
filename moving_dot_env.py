import numpy as np
import cv2


class MovingDotEnv:

    def __init__(self):

        # Environment parameters
        self.dot_size = [2, 2]
        self.random_start = True
        self.max_steps = 1000

        self.observation_space = np.array((210, 160, 3), dtype=np.uint8)
        self.centre = np.array([80, 105])

        self.reset()

    def reset(self):
        if self.random_start:
            x = np.random.randint(low=0, high=160)
            y = np.random.randint(low=0, high=210)
            self.pos = [x, y]
        else:
            self.pos = [0, 0]
        self.steps = 0
        ob = self._get_ob()
        return ob

    # This is important because for e.g. A3C each worker should be exploring
    # the environment differently, therefore seeds the random number generator
    # of each environment differently. (This influences the random start
    # location.)
    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def _get_ob(self):
        ob = np.zeros((210, 160, 3), dtype=np.uint8)
        x = self.pos[0]
        y = self.pos[1]
        w = self.dot_size[0]
        h = self.dot_size[1]
        ob[y - h:y + h, x - w:x + w, :] = 255
        return ob

    @staticmethod
    def get_action_meanings():
        return ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']

    def step(self, action):
        prev_pos = self.pos[:]

        self._update_pos(action)

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        return ob, reward, episode_over, {}

    def _update_pos(self, action):
        """ subclass is supposed to implement the logic
            to update the frame given an action at t """
        raise NotImplementedError

    # Based on gym's atari_env.py
    def render(self, mode='human', close=False):
        if close:
            cv2.destroyAllWindows()
            return

        img = self._get_ob()
        cv2.imshow("obs", img)
        cv2.waitKey(2)


class MovingDotDiscreteEnv(MovingDotEnv):
    """ Discrete Action MovingDot env """
    def __init__(self):
        super(MovingDotDiscreteEnv, self).__init__()
        # self.action_space = spaces.Discrete(5)

    def reset(self):
        return super(MovingDotDiscreteEnv, self).reset()

    def _update_pos(self, action):
        assert action >= 0 and action <= 4

        if action == 0:
            # NOOP
            pass
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            self.pos[0] -= 1
        self.pos[0] = np.clip(self.pos[0],
                              self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(self.pos[1],
                              self.dot_size[1], 209 - self.dot_size[1])
