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
        _, reward, done, info = test_env.step(a)
        test_env.env.render()
        time.sleep(0.005)
        print(f"reward: {reward}")
        print(info)
        if done:
            break
    test_env.close()
    exit(0)


if __name__ == '__main__':
    params = get_params()
    test_env = make_atari(params["env_name"])
    n_actions = test_env.action_space.n
    max_steps = 2000000  # test_env._max_episode_steps
    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{n_actions}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"])
    env.seed(123)

    stacked_frames = np.zeros(shape=[84, 84, 4], dtype=np.uint8)
    agent = Agent(n_actions=n_actions,
                  state_shape=[84, 84, 4],
                  **params)
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
        s = env.reset()
        stacked_frames = stack_frames(stacked_frames, s, True)
        episode_reward = 0
        episode_loss = 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, max_steps + 1):

            stacked_frames_copy = stacked_frames.copy()
            action = agent.choose_action(stacked_frames_copy)
            s_, r, d, _ = env.step(action)
            stacked_frames = stack_frames(stacked_frames, s_, False)
            r = np.clip(r, -1.0, 1.0)
            agent.store(stacked_frames_copy, action, r, stacked_frames, d)
            # env.render()
            # time.sleep(0.005)
            if step % params["train_period"] == 0:
                loss = agent.train()
                episode_loss += loss

            # if step % 5000:
            #     agent.hard_update_of_target_network()
            agent.soft_update_of_target_network()

            episode_reward += r
            if d:
                logger.off()
                if params["train_from_scratch"]:
                    agent.update_epsilon(episode)
                logger.log(episode, episode_reward, episode_loss, step)
                if episode % params["interval"] == 0:
                    logger.save_weights(episode)

                episode += 1
                s = env.reset()
                stacked_frames = stack_frames(stacked_frames, s, True)
                episode_reward = 0
                episode_loss = 0
                logger.on()

    else:
        episode = params["max_episodes"]
        step = max_steps
        # region play
        play_path = "./models/" + logger.dir + "/" "episode" + str(episode) + "-" + "step" + str(step)
        player = Play(agent, env, play_path)
        player.evaluate()
        # endregion

# Breakout showed sings of learning after 5000 episodes!!!!
