from torch import from_numpy
import torch
from Brain.model import Model
from torch.optim.adam import Adam
import numpy as np
from Memory.replay_memory import ReplayMemory, Transition
from collections import deque


class Agent:
    def __init__(self, **config):
        self.config = config
        self.n_actions = self.config["n_actions"]
        self.state_shape = self.config["state_shape"]
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.initial_mem_size_to_train = self.config["initial_mem_size_to_train"]
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.memory = ReplayMemory(self.config["mem_size"], self.config["alpha"])
        self.v_min = self.config["v_min"]
        self.v_max = self.config["v_max"]
        self.n_atoms = self.config["n_atoms"]
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long() \
            .unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)

        self.nstep = self.config["n_step"]
        self.nstep_buffer = deque(maxlen=self.nstep)

        self.online_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model = Model(self.state_shape, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.hard_update_of_target_network()

        self.optimizer = Adam(self.online_model.parameters(), lr=self.config["lr"], eps=self.config["adam_eps"])

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).byte().to(self.device)
        with torch.no_grad():
            action = self.online_model.get_q_value(state.permute(dims=[0, 3, 1, 2])).argmax(-1).item()

        return action

    def store(self, state, action, reward, next_state, done):
        """Save I/O s to store them in RAM and not to push pressure on GPU RAM """
        assert state.dtype == "uint8"
        assert next_state.dtype == "uint8"
        assert reward % 1 == 0, "Reward isn't an integer number so change the type it's stored in the replay memory."
        self.nstep_buffer.append((state, action, reward, next_state, done))
        if len(self.nstep_buffer) < self.nstep:
            return

        reward, next_state, done = self.get_nstep_returns()
        state, action, _, _, _ = self.nstep_buffer.pop()

        state = from_numpy(state).byte().to("cpu")
        reward = torch.CharTensor([reward])
        action = torch.ByteTensor([action]).to('cpu')
        next_state = from_numpy(next_state).byte().to('cpu')
        done = torch.BoolTensor([done])
        self.memory.add(state, action, reward, next_state, done)

    def soft_update_of_target_network(self, tau=0.001):
        for target_param, local_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.target_model.eval()

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

    def train(self, beta):
        if len(self.memory) < self.initial_mem_size_to_train:
            return 0  # as no loss
        batch, weights, indices = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)
        weights = from_numpy(weights).float().to(self.device)

        with torch.no_grad():
            q_eval_next = self.online_model.get_q_value(next_states)
            selected_actions = torch.argmax(q_eval_next, dim=-1)
            q_next = self.target_model(next_states)[range(self.batch_size), selected_actions]

            projected_atoms = rewards + (self.gamma ** self.nstep) * self.support * (1 - dones.byte())
            projected_atoms = projected_atoms.clamp(self.v_min, self.v_max)

            b = (projected_atoms - self.v_min) / self.delta_z
            lower_bound = b.floor().long()
            upper_bound = b.ceil().long()
            lower_bound[(upper_bound > 0) * (lower_bound == upper_bound)] -= 1
            upper_bound[(lower_bound < (self.n_atoms - 1)) * (lower_bound == upper_bound)] += 1

            projected_dist = torch.zeros(q_next.size()).to(self.device)
            projected_dist.view(-1).index_add_(0, (lower_bound + self.offset).view(-1),
                                               (q_next * (upper_bound.float() - b)).view(-1))
            projected_dist.view(-1).index_add_(0, (upper_bound + self.offset).view(-1),
                                               (q_next * (b - lower_bound.float())).view(-1))

        eval_dist = self.online_model(states)[range(self.batch_size), actions.squeeze().long()]
        dqn_loss = - (projected_dist * torch.log(eval_dist)).sum(-1)
        td_error = dqn_loss.abs() + 1e-6
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        dqn_loss = (dqn_loss * weights).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 10.0)
        self.optimizer.step()

        self.online_model.reset()
        self.target_model.reset()
        return dqn_loss.detach().cpu().numpy()

    def ready_to_play(self, state_dict):
        self.online_model.load_state_dict(state_dict)
        self.online_model.eval()

    def get_nstep_returns(self):
        reward, next_state, done = self.nstep_buffer[-1][-3:]

        for transition in reversed(list(self.nstep_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done
