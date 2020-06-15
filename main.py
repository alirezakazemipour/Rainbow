import gym
from logger import Logger
from play import Play
from agent import Agent
from utils import *
from config import get_params

params = get_params()

test_env = gym.make(params["env_name"])
n_actions = test_env.action_space.n
max_steps = test_env._max_episode_steps

save_interval = params["save_interval"]

logger = Logger()

if __name__ == '__main__':

    env = gym.make(params["env_name"])
    stacked_frames = np.zeros(shape=[84, 84, 4], dtype='float32')
    agent = Agent(n_actions=n_actions,
                  state_shape=[84, 84, 4],
                  **params)
    if params["do_train"]:

        for episode in range(1, params["max_episodes"] + 1):
            s = env.reset()
            stacked_frames = stack_frames(stacked_frames, s, True)
            episode_reward = 0
            episode_loss = 0

            logger.on()

            for step in range(1, max_steps + 1):

                stacked_frames_copy = stacked_frames.copy()
                action = agent.choose_action(stacked_frames_copy)
                for _ in range(4):
                    s_, r, d, _ = env.step(action)
                stacked_frames = stack_frames(stacked_frames, s_, False)
                r = np.clip(r, -1.0, 1.0)
                agent.store(stacked_frames_copy, action, r, stacked_frames, d)
                # env.render()
                loss = agent.train()
                episode_loss += loss

                episode_reward += r

                # if step % save_interval == 0:
                #     episode_log.save_weights(agent.eval_model, agent.optimizer, episode, step)
                if d:
                    break

            logger.off()
            logger.print(episode, episode_reward, episode_loss, step, len(agent.memory), agent.epsilon, **params)
            agent.update_epsilon()
                
    else:
        episode = params["max_episodes"]
        step = max_steps
        # region play
        play_path = "./models/" + logger.dir + "/" "episode" + str(episode) + "-" + "step" + str(step)
        player = Play(agent, env, play_path)
        player.evaluate()
        # endregion
