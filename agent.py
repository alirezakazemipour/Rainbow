from torch import nn, from_numpy
import torch
from model import Model
from torch.optim.adam import Adam
from logger import Logger
import numpy as np
from replay_memory import ReplayMemory, Transition

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()


class Agent:
    def __init__(self, n_actions, state_shape, **config):
        self.config = config
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.epsilon = self.config["epsilon"]
        self.decay_rate = self.config["decay_rate"]
        self.min_epsilon = self.config["min_epsilon"]
        self.memory = ReplayMemory(self.config["mem_size"])

        self.online_model = Model(self.state_shape, self.n_actions).to(self.device)
        self.target_model = Model(self.state_shape, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.optimizer = Adam(self.online_model.parameters(), lr=self.config["lr"], eps=self.config["adam_eps"])
        self.loss_fn = nn.MSELoss()

        self.update_counter = 0

    def choose_action(self, state):

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).byte().to(self.device)
            with torch.no_grad():
                action = self.online_model(state.permute(dims=[0, 3, 1, 2])).argmax(-1).item()

        Logger.simulation_steps += 1
        return action

    def store(self, state, action, reward, next_state, done):
        """Save I/O s to store them in RAM and not to push pressure on GPU RAM """

        state = from_numpy(state).byte().to("cpu")
        reward = torch.CharTensor([reward])
        action = torch.ByteTensor([action]).to('cpu')
        next_state = from_numpy(next_state).byte().to('cpu')
        done = torch.BoolTensor([done])
        self.memory.add(state, action, reward, next_state, done)

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        target_model.eval()

    def hard_update_of_target_network(self):
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.config["batch_size"], *self.state_shape)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device).view((-1, 1))
        next_states = torch.cat(batch.next_state).to(self.device).view(self.config["batch_size"], *self.state_shape)
        dones = torch.cat(batch.done).to(self.device).view((-1, 1))
        states = states.permute(dims=[0, 3, 1, 2])
        actions = actions.view((-1, 1))
        next_states = next_states.permute(dims=[0, 3, 1, 2])
        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.memory) < 10000:
            return 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        q_eval = self.online_model(states).gather(dim=-1, index=actions.long())
        with torch.no_grad():
            q_next = self.target_model(next_states)
            q_eval_next = self.online_model(next_states)

            next_actions = q_eval_next.argmax(dim=-1).view(-1, 1)
            q_next = q_next.gather(dim=-1, index=next_actions)

            q_target = rewards + self.gamma * q_next * (1 - dones.byte())
        dqn_loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 10.0)
        self.optimizer.step()

        self.soft_update_of_target_network(self.online_model, self.target_model, self.tau)
        # if self.update_counter % 5000 == 0:
        #     self.hard_update_of_target_network()

        self.update_counter += 1
        return dqn_loss.detach().cpu().numpy()

    def ready_to_play(self, path):
        model_state_dict, _ = Logger.load_weights(path)
        self.online_model.load_state_dict(model_state_dict)
        self.online_model.eval()

    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-episode * self.decay_rate)  # \
        # if len(self.memory) > 10000 else 1.0
