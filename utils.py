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


def step_repetitive_action(env, max_lives, action):
    successive_states = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    reward, done = 0, False
    for t in range(4):
        state, r, done, info = env.step(action)
        reward += r
        if t == 2:
            successive_states[0] = state
        elif t == 3:
            successive_states[1] = state
        if done:
            break

    state = successive_states.max(axis=0)
    if max_lives > info["ale.lives"] > 0:
        done = True

    return state, reward, done, info
