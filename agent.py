from torch import nn, from_numpy
import torch
from model import Model
from torch.optim.adam import Adam
from logger import Logger
import numpy as np
from replay_memory import ReplayMemory, Transition
from collections import deque

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
        # self.epsilon = self.config["epsilon"]
        # self.decay_rate = self.config["decay_rate"]
        # self.min_epsilon = self.config["min_epsilon"]
        self.memory = ReplayMemory(self.config["mem_size"], self.config["alpha"])

        self.v_min = self.config["V_min"]
        self.v_max = self.config["V_max"]
        self.n_atoms = self.config["N_atoms"]
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.online_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.optimizer = Adam(self.online_model.parameters(), lr=self.config["lr"], eps=self.config["adam_eps"])
        self.loss_fn = nn.MSELoss()

        self.steps = 0

        self.n_step_buffer = deque(maxlen=self.config["multi_step_n"])

    def choose_action(self, state):

        # if np.random.random() < self.epsilon:
        #     action = np.random.randint(0, self.n_actions)
        # else:
        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.online_model.get_q_value(state.permute(dims=[0, 3, 2, 1])).argmax(-1).item()

        self.steps += 1
        Logger.simulation_steps += 1

        return action

    def store(self, state, action, reward, next_state, done):
        """Save I/O s to store them in RAM and not to push pressure on GPU RAM """

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.config["multi_step_n"]:
            return
        reward, next_state, done = self.n_step_returns()
        state, action, _, _, _ = self.n_step_buffer.pop()

        state = from_numpy(state).float().to('cpu')
        reward = torch.Tensor([reward])
        action = torch.Tensor([action]).to('cpu')
        next_state = from_numpy(next_state).float().to('cpu')
        done = torch.Tensor([done])
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
        states = states.permute(dims=[0, 3, 2, 1])
        actions = actions.view((-1, 1))
        next_states = next_states.permute(dims=[0, 3, 2, 1])
        return states, actions, rewards, next_states, dones

    def train(self, beta):
        if len(self.memory) < 1000: # to reduce correlation
            return 0  # as no loss
        batch, weights, indices = self.memory.sample(self.batch_size, beta)
        weights = from_numpy(weights).float().to(self.device)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        with torch.no_grad():
            q_eval_next = self.online_model.get_q_value(next_states)
            next_actions = q_eval_next.argmax(dim=-1)
            q_next = self.target_model(next_states)[range(self.batch_size), next_actions.long()]

            projected_atoms = rewards + (self.config["gamma"] ** self.config["multi_step_n"]) * self.support * (1 - dones)
            projected_atoms = projected_atoms.clamp_(self.v_min, self.v_max)

            b = (projected_atoms - self.v_min) / self.delta_z
            lower_bound = b.floor().long()
            upper_bound = b.ceil().long()

            # projected_dist = torch.zeros((self.batch_size, self.n_atoms)).to(self.device)
            # for i in range(self.batch_size):
            #     for j in range(self.n_atoms):
            #         projected_dist[i, lower_bound[i, j]] += (q_next * (upper_bound - b))[i, j]
            #         projected_dist[i, upper_bound[i, j]] += (q_next * (b - lower_bound))[i, j]

            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long() \
                .unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)

            projected_dist = torch.zeros(q_next.size()).to(self.device)
            projected_dist.view(-1).index_add_(0, (lower_bound + offset).view(-1),
                                               (q_next * (upper_bound.float() - b)).view(-1))
            projected_dist.view(-1).index_add_(0, (upper_bound + offset).view(-1),
                                               (q_next * (b - lower_bound.float())).view(-1))

        eval_dist = self.online_model(states)[range(self.batch_size), actions.squeeze().long()]
        dqn_loss = - (projected_dist * torch.log(eval_dist + 1e-8)).sum(-1)
        td_error = dqn_loss.abs()
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy() + 1e-6)
        dqn_loss = (dqn_loss * weights).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 10.0)
        self.optimizer.step()

        # self.soft_update_of_target_network(self.online_model, self.target_model, self.tau)
        if self.steps % 8000 == 0:
            self.hard_update_of_target_network()

        self.online_model.reset()
        self.target_model.reset()
        return dqn_loss.detach().cpu().numpy()

    def ready_to_play(self, path):
        model_state_dict, _ = Logger.load_weights(path)
        self.online_model.load_state_dict(model_state_dict)
        self.online_model.eval()

    # def update_epsilon(self):
    #     self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.min_epsilon + self.decay_rate \
    #         else self.min_epsilon

    def n_step_returns(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done
