from skimage.transform import resize
import numpy as np


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
