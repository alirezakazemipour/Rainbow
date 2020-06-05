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

TRAIN_FROM_SCRATCH = True


class Agent:
    def __init__(self, n_actions, state_shape, epsilon_start, epsilon_end,
                 epsilon_decay, **config):
        self.n_actions = n_actions
        self.config = config
        self.state_shape = state_shape
        self.update_count = 0
        self.eps_threshold = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_model = Model(self.state_shape, self.n_actions).to(self.device)
        self.target_model = Model(self.state_shape, self.n_actions).to(self.device)

        if not TRAIN_FROM_SCRATCH:
            # TODO
            # Load weights and other params
            pass

        self.loss_fn = nn.MSELoss()
        # self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()  # Sets batchnorm and droupout for evaluation not training
        self.optimizer = Adam(self.eval_model.parameters(), lr=self.config["lr"])
        self.memory = ReplayMemory(self.config["mem_size"])

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start

        self.steps = 0
        self.multi_step_buffer = []

    def choose_action(self, state):

        if np.random.random() > self.eps_threshold:
            with torch.no_grad():
                state = torch.unsqueeze(from_numpy(state).float().to(self.device), dim=0)
                action = self.eval_model(
                    state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0]

        else:
            action = torch.randint(low=0, high=self.n_actions, size=(1,), device=self.device)[0]
        self.steps += 1
        Logger.simulation_steps += 1

        return action

    def get_action(self, state):
        state = from_numpy(state).float().to(self.device)
        state = torch.unsqueeze(state, dim=0)
        return self.eval_model(state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0]

    def store(self, state, action, reward, next_state, done):
        """Save I/O s to store them in RAM and not to push pressure on GPU RAM """

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
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.config["batch_size"], *self.state_shape)
        dones = torch.cat(batch.done).to(self.device)
        states = states.permute(dims=[0, 3, 2, 1])
        actions = actions.view((-1, 1))
        next_states = next_states.permute(dims=[0, 3, 2, 1])
        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.memory) < self.config["batch_size"]:
            return 0  # as no loss
        batch = self.memory.sample(self.config["batch_size"])
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        x = states
        q_eval = self.eval_model(x).gather(dim=1, index=actions)
        with torch.no_grad():
            q_next = self.target_model(next_states).detach()

            q_eval_next = self.eval_model(next_states).detach()
            max_action = torch.argmax(q_eval_next, dim=-1)

            batch_indices = torch.arange(end=self.config["batch_size"], dtype=torch.int32)
            target_value = q_next[batch_indices.long(), max_action] * (1 - dones)

            q_target = rewards + self.config["gamma"] * target_value
        loss = self.loss_fn(q_eval, q_target.view(self.config["batch_size"], 1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 10)

        # for param in self.Qnet.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.update_count += 1
        var = loss.detach().cpu().numpy()
        self.soft_update_of_target_network(self.eval_model, self.target_model)

        return var

    def ready_to_play(self, path):
        model_state_dict, _ = Logger.load_weights(path)
        self.eval_model.load_state_dict(model_state_dict)
        self.eval_model.eval()

    def update_epsilon(self):
        self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                             np.exp(-1. * self.steps / self.epsilon_decay)

    def multi_step_returns(self, state, action, reward, nex_state):
        self.multi_step_buffer.append((state, action, reward, nex_state))

        if len(self.multi_step_buffer) < self.config["multi_step_n"]:
            return

        R = sum(
            [self.multi_step_buffer[i][2] * (self.config["gamma"] ** i) for i in range(self.config["multi_step_n"])])
        state, action, _, _ = self.multi_step_buffer.pop(0)
        # print("R:", R)
        return state, action, R, nex_state

