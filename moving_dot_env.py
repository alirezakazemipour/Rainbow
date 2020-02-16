import numpy as np
import cv2


class MovingDotEnv:
    CV_ACTION = {0: np.array([0, -1]),
                 1: np.array([-1, 0]),
                 2: np.array([0, 1]),
                 3: np.array([1, 0])}

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
        # prev_pos = self.pos[:] # knock knock
        action = int(action)
        prev_pos = self.pos

        # self._update_pos(action)  # knock knock
        self.pos = self._update_pos(self.pos, action)

        # ob = self._get_ob() # knock knock

        old_dis = np.linalg.norm(prev_pos - self.centre)
        new_dis = np.linalg.norm(self.pos - self.centre)
        if old_dis > new_dis:
            reward = 1
        else:
            reward = -1

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        self.pose = self.check_obstacles(self.pos)

        reach_goal = self.check_terminal(self.pose)

        # dist1 = np.linalg.norm(prev_pos - self.centre)
        # dist2 = np.linalg.norm(self.pos - self.centre)
        # if dist2 < dist1:
        #     reward = 1
        # elif dist2 == dist1:
        #     reward = 0
        # else:
        #     reward = -1
        ob = self._get_ob()

        return ob, reward, episode_over or reach_goal, {}

    def _update_pos(self, old_pos, action):  # Knock Knock
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

    def check_obstacles(self, new_pos):
        """
         We consider the outer pixels of environment as obstacles and agent is supposed to prevent from entering those
         pixels so if it does it will be returned to pixed it was before taking the action and there will be a punishment
         for it if anything has been specified for obstacle reward.
        """
        (x1, y1)= new_pos
        if x1 < 0:
            delta = abs(x1)
            x1 += delta
            # x2 += delta
        elif x1 >= 160:
            delta = x1 - (160 - 1)
            x1 -= delta
            # x2 -= delta

        if y1 < 0:
            delta = abs(y1)
            y1 += delta
            # y2 += delta

        elif y1 >= 210:
            delta = y1 - (210 - 1)
            y1 -= delta
            # y2 -= delta
        new_pos = np.array([x1, y1])
        return new_pos

    def check_terminal(self, pos):
        dis = np.sqrt(np.sum(np.square(np.array(np.mean(pos, axis=0)) - np.array(self.centre))))
        if dis < 5:
            return True
        return False


class MovingDotDiscreteEnv(MovingDotEnv):
    """ Discrete Action MovingDot env """
    def __init__(self):
        super(MovingDotDiscreteEnv, self).__init__()
        # self.action_space = spaces.Discrete(5)

    def reset(self):
        return super(MovingDotDiscreteEnv, self).reset()

    def _update_pos(self, old_pos, action):
        assert action >= 0 and action <= 3

        p1, p2 = old_pos
        x = p1 + self.CV_ACTION[action][0] * 1
        y = p2 + self.CV_ACTION[action][1] * 1
        new_pos = (x, y)
        return new_pos
