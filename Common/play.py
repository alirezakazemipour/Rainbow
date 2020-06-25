from Common.utils import *
import os

class Play:
    def __init__(self, agent, env, weights, **config):
        self.config = config
        self.agent = agent
        self.weights = weights
        self.agent.ready_to_play(self.weights)
        self.env = env
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + self.config["algo"] + ".avi", self.fourcc, 50.0, (160, 210))

    def evaluate(self):
        stacked_states = np.zeros(shape=[84, 84, 4], dtype=np.uint8)
        total_reward = 0
        print("--------Play mode--------")
        for _ in range(1):
            done = 0
            state = self.env.reset()
            episode_reward = 0
            stacked_states = stack_states(stacked_states, state, True)

            while not done:
                stacked_frames_copy = stacked_states.copy()
                action = self.agent.choose_action(stacked_frames_copy)
                next_state, r, done, _ = self.env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                self.env.render()
                episode_reward += r
                self.VideoWriter.write(cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))
            total_reward += episode_reward

        print("Total episode reward:", total_reward)
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()
