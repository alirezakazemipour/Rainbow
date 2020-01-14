import gym_moving_dot
import gym
import numpy as np
from skimage.transform import resize
from logger import LOG

from agent import Agent

ENV_NAME = "MovingDotDiscrete-v0"
# ENV_NAME = "MontezumaRevenge-v0"
MAX_EPISODES = 10

episode_log = LOG()


def rgb2gray(img):
    return 0.2125 * img[..., 0] + 0.7154 * img[..., 1] + 0.0721 * img[..., 2]


def preprocessing(img):
    img = rgb2gray(img) / 255.0
    img = resize(img, output_shape=[84, 84])
    return img


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=2)
    else:
        stacked_frames = stacked_frames[..., :3]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=2)], axis=2)
    return stacked_frames


if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    n_actions = env.action_space.n
    stacked_frames = np.zeros(shape=[84, 84, 4], dtype='float32')
    agent = Agent(n_actions, 0.99, 0.005, 0.001, [84, 84, 4], 10000, alpha=0.99, epsilon_start=0.9, epsilon_end=0.05,
                  epsilon_decay=200, batch_size=512)

    for episode in range(1, MAX_EPISODES + 1):
        s = env.reset()
        stacked_frames = stack_frames(stacked_frames, s, True)
        episode_reward = 0
        episode_loss = 0

        episode_log.on()

        for step in range(1, 500 + 1):
            # env.render()

            stacked_frames_copy = stacked_frames.copy()
            action = agent.choose_action(stacked_frames_copy)
            s_, r, d, _ = env.step(action)
            stacked_frames = stack_frames(stacked_frames, s_, False)
            agent.store(stacked_frames_copy, action, r, stacked_frames, d)
            # env.render()
            if d:
                break

            loss = agent.train()
            episode_reward += r
            episode_loss += loss

        episode_log.off()
        episode_log.printer(episode, episode_reward, episode_loss, agent.eps_threshold, step)
        # print(f'episode: {episode}. reward: {episode_reward}. loss: {episode_loss}')
