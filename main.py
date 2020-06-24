from logger import Logger
from play import Play
from agent import Agent
from utils import *
from config import get_params
import time


def intro_env():
    test_env.reset()
    for _ in range(max_steps):
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


if __name__ == '__main__':
    params = get_params()
    test_env = make_atari(params["env_name"])
    n_actions = test_env.action_space.n
    max_steps = int(10e7)  # test_env._max_episode_steps
    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{n_actions}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"])
    env.seed(123)

    agent = Agent(n_actions=n_actions, state_shape=[84, 84, 4], **params)
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

        # for episode in range(min_episode + 1, params["max_episodes"] + 1):
        stacked_states = np.zeros(shape=[84, 84, 4], dtype=np.uint8)
        state = env.reset()
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        loss = 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, max_steps + 1):

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
                beta = min(1.0, params["beta"] + episode * (1.0 - params["beta"]) / 1500)
                loss += agent.train(beta)
            agent.soft_update_of_target_network()
            # if step % 5000:
            #     agent.hard_update_of_target_network()

            if done:
                logger.off()
                if params["train_from_scratch"]:
                    agent.update_epsilon(episode)
                logger.log(episode, episode_reward, loss, step)

                episode += 1
                state = env.reset()
                stacked_frames = stack_states(stacked_states, state, True)
                episode_reward = 0
                episode_loss = 0
                logger.on()

    else:
        # region play
        chekpoint = logger.load_weights()
        player = Play(agent, env, chekpoint["online_model_state_dict"])
        player.evaluate()
        # endregion

# Breakout showed sings of learning after 5000 episodes!!!!
