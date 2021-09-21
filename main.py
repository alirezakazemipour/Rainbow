from comet_ml import Experiment
from Common import Logger, Play, get_params, make_atari, make_state
from Brain import Agent
from collections import namedtuple
import time
import numpy as np
import os
import math


# region introduction to env.
def intro_env():
    for e in range(5):
        test_env.reset()
        for _ in range(test_env._max_episode_steps):
            a = test_env.env.action_space.sample()
            _, r, d, info = test_env.step(a)
            test_env.env.render()
            time.sleep(0.005)
            print(f"reward: {r}")
            print(info)
            if d:
                break
    test_env.close()
    exit(0)


# endregion

if __name__ == '__main__':
    params = get_params()
    os.environ["PYTHONHASHSEED"] = str(params["seed"])

    test_env = make_atari(params["env_name"], params["seed"])
    params.update({"n_actions": test_env.action_space.n})
    del test_env
    params.update({"transition": namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))})

    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{params['n_actions']}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"], params["seed"])

    agent = Agent(**params)
    experiment = Experiment()
    logger = Logger(agent, experiment=experiment, **params)

    if not params["train_from_scratch"]:
        chekpoint = logger.load_weights()
        agent.online_model.load_state_dict(chekpoint["online_model_state_dict"])
        agent.hard_update_of_target_network()
        params.update({"beta": chekpoint["beta"]})
        min_episode = chekpoint["episode"]

        print("Keep training from previous run.")
    else:
        min_episode = 0
        print("Train from scratch.")

    if params["do_train"]:

        sign = lambda x: bool(x > 0) - bool(x < 0)
        state = np.zeros(shape=params["state_shape"], dtype=np.uint8)
        obs = env.reset()
        state = make_state(state, obs, True)
        episode_reward = 0
        episode_len = 0
        episode_loss , episode_g_norm = 0, 0
        beta = params["beta"]
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):
            episode_len += 1
            action = agent.choose_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = make_state(state, next_obs, False)
            r = sign(reward)
            agent.store(state, action, r, next_state, done)
            episode_reward += reward

            if step % params["train_period"] == 0:
                beta = min(1.0, params["beta"] + step * (1.0 - params["beta"]) / params["final_annealing_beta_steps"])
                loss, g_norm = agent.train(beta)
            else:
                loss, g_norm = 0, 0
            agent.soft_update_of_target_network(params["tau"])
            episode_loss += loss
            episode_g_norm = g_norm
            state = next_state
            if done:
                logger.off()
                logger.log(episode,
                           episode_reward,
                           episode_loss / episode_len * params["train_period"],
                           episode_g_norm / episode_len * params["train_period"],
                           step,
                           beta)

                episode += 1
                obs = env.reset()
                state = make_state(state, obs, True)
                episode_reward = 0
                episode_len = 0
                episode_loss = 0
                episode_g_norm = 0
                logger.on()

    else:
        # region play
        chekpoint = logger.load_weights()
        player = Play(agent, env, chekpoint["online_model_state_dict"], **params)
        player.evaluate()
        # endregion
