from Common.logger import Logger
from Common.play import Play
from Brain.agent import Agent
from Common.utils import *
from Common.config import get_params
import time


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

    test_env = make_atari(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})

    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{params['n_actions']}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"])
    env.seed(123)

    agent = Agent(**params)
    logger = Logger(agent, **params)

    if not params["train_from_scratch"]:
        chekpoint = logger.load_weights()
        agent.online_model.load_state_dict(chekpoint["online_model_state_dict"])
        agent.hard_update_of_target_network()
        agent.epsilon = chekpoint["epsilon"]
        min_episode = chekpoint["episode"]

        print("Keep training from previous run.")
    else:
        min_episode = 0
        print("Train from scratch.")

    if params["do_train"]:

        stacked_states = np.zeros(shape=params["state_shape"], dtype=np.uint8)
        state = env.reset()
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        beta = params["beta"]
        loss = 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):

            stacked_states_copy = stacked_states.copy()
            action = agent.choose_action(stacked_states_copy)
            next_state, reward, done, _ = env.step(action)
            stacked_states = stack_states(stacked_states, next_state, False)
            reward = np.clip(reward, -1.0, 1.0)
            agent.store(stacked_states_copy, action, reward, stacked_states, done)
            episode_reward += reward

            # env.render()
            # time.sleep(0.005)
            if step % params["train_period"] == 0:
                beta = min(1.0, params["beta"] + step * (1.0 - params["beta"]) / params["final_annealing_beta_steps"])
                loss += agent.train(beta)
            agent.soft_update_of_target_network()
            # if step % 5000:
            #     agent.hard_update_of_target_network()

            if done:
                logger.off()
                if params["train_from_scratch"]:
                    agent.update_epsilon(episode)
                logger.log(episode, episode_reward, loss, step, beta)

                episode += 1
                state = env.reset()
                stacked_frames = stack_states(stacked_states, state, True)
                episode_reward = 0
                episode_loss = 0
                logger.on()

    else:
        # region play
        chekpoint = logger.load_weights()
        player = Play(agent, env, chekpoint["online_model_state_dict"], **params)
        player.evaluate()
        # endregion

