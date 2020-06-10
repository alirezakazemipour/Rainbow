from torch import nn, from_numpy
import torch
from model import Model
from torch.optim.adam import Adam
from logger import Logger
import numpy as np
from collections import deque
from replay_memory import ReplayMemory, Transition

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

TRAIN_FROM_SCRATCH = True


class Agent:
    def __init__(self, n_actions, state_shape, **config):
        self.n_actions = n_actions
        self.config = config
        self.batch_size = self.config["batch_size"]
        self.state_shape = state_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_step_buffer = deque(maxlen=self.config["multi_step_n"])
        self.v_min = self.config["V_min"]
        self.v_max = self.config["V_max"]
        self.n_atoms = self.config["N_atoms"]
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.eval_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()  # Sets batchnorm and droupout for evaluation not training

        if not TRAIN_FROM_SCRATCH:
            # TODO
            # Load weights and other params
            pass

        self.optimizer = Adam(self.eval_model.parameters(), lr=self.config["lr"], eps=self.config["adam_eps"])
        self.memory = ReplayMemory(self.config["mem_size"])

        self.steps = 0
        self.multi_step_buffer = deque(maxlen=self.config["multi_step_n"])

    def choose_action(self, state):

        with torch.no_grad():
            state = torch.unsqueeze(from_numpy(state).float().to(self.device), dim=0)
            action = self.eval_model.get_q_value(
                state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0]

        self.steps += 1
        Logger.simulation_steps += 1

        return action

    def get_action(self, state):
        state = from_numpy(state).float().to(self.device)
        state = torch.unsqueeze(state, dim=0)
        return self.eval_model.get_q_value(state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0]

    def store(self, state, action, reward, next_state, done):
        """Save I/O s to store them in RAM and not to push pressure on GPU RAM """

        self.multi_step_buffer.append((state, action, reward, next_state, done))
        if len(self.multi_step_buffer) < self.config["multi_step_n"]:
            return

        reward, next_state, done = self.multi_step_returns()
        state, action, _, _, _ = self.multi_step_buffer.pop()

        state = from_numpy(state).float().to('cpu')
        reward = torch.Tensor([reward])
        action = torch.unsqueeze(action, dim=0)
        next_state = from_numpy(next_state).float().to('cpu')
        done = torch.Tensor([done])
        self.memory.push(state, action.to('cpu'), reward, next_state, done)

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update_of_target_network(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())
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

    def train(self):
        if len(self.memory) < self.config["batch_size"]:
            return 0  # as no loss
        batch = self.memory.sample(self.config["batch_size"])
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        with torch.no_grad():
            q_eval_next = self.eval_model.get_q_value(next_states)
            next_actions = q_eval_next.argmax(dim=-1)
            q_next = self.target_model(next_states)[range(self.batch_size), next_actions.long()]

            projected_atoms = rewards + (self.config["gamma"] ** self.config["multi_step_n"]) * self.support * (
                        1 - dones)
            projected_atoms = projected_atoms.clamp_(self.v_min, self.v_max)

            b = (projected_atoms - self.v_min) / self.delta_z
            lower_bound = b.floor().long()
            upper_bound = b.ceil().long()

            projected_dist = torch.zeros((self.batch_size, self.n_atoms)).to(self.device)
            for i in range(self.batch_size):
                for j in range(self.n_atoms):
                    projected_dist[i, lower_bound[i, j]] += (q_next * (upper_bound - b))[i, j]
                    projected_dist[i, upper_bound[i, j]] += (q_next * (b - lower_bound))[i, j]

        eval_dist = self.eval_model(states)[range(self.batch_size), actions.squeeze().long()]
        dqn_loss = - (projected_dist * torch.log(eval_dist)).sum(-1).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 10)

        # for param in self.Qnet.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        var = dqn_loss.detach().cpu().numpy()

        self.target_model.reset()
        self.eval_model.reset()

        # self.soft_update_of_target_network(self.eval_model, self.target_model, self.config["tau"])
        if self.steps % 1000 == 0:
            self.hard_update_of_target_network()

        return var

    def ready_to_play(self, path):
        model_state_dict, _ = Logger.load_weights(path)
        self.eval_model.load_state_dict(model_state_dict)
        self.eval_model.eval()

    def multi_step_returns(self):

        reward, next_state, done = self.multi_step_buffer[-1][-3:]

        for transition in reversed(list(self.multi_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.config["gamma"] * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done
