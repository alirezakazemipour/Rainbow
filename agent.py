from torch import nn, from_numpy
import torch
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from logger import LOG

import numpy as np

from replay_memory import ReplayMemory, Transition

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

TRAIN_FROM_SCRATCH = True


def get_conv_out(n, stride, kernel_size, padding=0):
    return (n + 2 * padding - kernel_size) // stride + 1


def initial_weights(module, linear_initialization=False):
    """
    check out https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L118
    """
    for m in module:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            # we can set bias to zero but as reference resnet initialization did, let it be as it is(default)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and linear_initialization:
            # it's better we don't initialize linear layers because our
            # linear layer has no activation function and it would be ok to go with default(not atari)
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class DQN(nn.Module):
    def __init__(self, n_actions, state_shape, name=''):
        super(DQN, self).__init__()
        self.model_name = name
        self.n_actions = n_actions
        width, height, channels = state_shape

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.resnet1 = ResNetLayer(16, filters=16, name="resnet1")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.resnet2 = ResNetLayer(32, 32, name="resnet2")

        convw = get_conv_out(width, kernel_size=2, padding=0, stride=2)  # Max pooling kernel_size and stride
        convh = get_conv_out(height, kernel_size=2, padding=0, stride=2)  # Max pooling kernel_size and stride

        self.fc = nn.Linear(convw * convh * 32, 512)
        nn.init.kaiming_normal_(self.fc.weight)
        self.fc.bias.data.zero_()
        self.s_a_value = nn.Linear(512, self.n_actions)

        initial_weights(self.modules())

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resnet1(x)
        x = F.relu(self.conv2(x))
        x = self.resnet2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.s_a_value(x)

        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, filters, name, strides=1, kernel_size=3):
        super(ResNetLayer, self).__init__()

        self.filters = filters
        self.in_channels = in_channels
        self.layer_name = name
        self.strides = strides
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.filters,
                               kernel_size=self.kernel_size,
                               stride=self.strides,
                               padding=1)

        self.batch_norm = nn.BatchNorm2d(self.filters)

        self.conv2 = nn.Conv2d(in_channels=self.filters,
                               out_channels=self.filters,
                               kernel_size=self.kernel_size,
                               stride=self.strides,
                               padding=1)
        initial_weights(self.modules())

    def forward(self, x):
        inputs = x
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = self.conv2(x)
        return F.relu(x + inputs)




class Agent:
    def __init__(self, n_actions, gamma, tau, lr, state_shape, capacity, alpha, epsilon_start, epsilon_end,
                 epsilon_decay, batch_size):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.state_shape = state_shape
        self.batch_size = batch_size

        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_model = DQN(name="eval_model", state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.target_model = DQN(name="target_model", state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)

        if not TRAIN_FROM_SCRATCH:
            # TODO
            # Load weights and other params
            pass

        self.loss_fn = nn.MSELoss()
        # self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()  # Sets batchnorm and droupout for evaluation not training
        # self.optimizer = RMSprop(self.eval_model.parameters(), lr=self.lr, alpha=alpha)
        self.optimizer = Adam(self.eval_model.parameters(), lr=self.lr)
        self.memory = ReplayMemory(capacity)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start

        self.steps = 0

    def choose_action(self, state):
        self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                             np.exp(-1. * self.steps / self.epsilon_decay)

        if np.random.random() > self.eps_threshold:
            with torch.no_grad():
                state = torch.unsqueeze(from_numpy(state).float().to(self.device), dim=0)
                action = self.eval_model(
                    state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0]

        else:
            action = torch.randint(low=0, high=self.n_actions, size=(1,), device=self.device)[0]
        self.steps += 1
        LOG.simulation_steps += 1

        return action

    def get_action(self, state):
        state = from_numpy(state).float().to(self.device)
        state = torch.unsqueeze(state, dim=0)
        return self.eval_model(state.permute(dims=[0, 3, 2, 1])).argmax(dim=1)[0].item()

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

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, *self.state_shape)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, *self.state_shape)
        dones = torch.cat(batch.done).to(self.device)
        states = states.permute(dims=[0, 3, 2, 1])
        actions = actions.view((-1, 1))
        next_states = next_states.permute(dims=[0, 3, 2, 1])
        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        x = states
        q_eval = self.eval_model(x).gather(dim=1, index=actions)
        with torch.no_grad():
            q_next = self.target_model(next_states)

            q_eval_next = self.eval_model(next_states)
            max_action = torch.argmax(q_eval_next, dim=-1)

            batch_indices = torch.arange(end=self.batch_size, dtype=torch.int32)
            target_value = q_next[batch_indices.long(), max_action] * (1 - dones)

            q_target = rewards + self.gamma * target_value
        loss = self.loss_fn(q_eval, q_target.view(self.batch_size, 1))

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 10)  # clip gradients to help stabilise training

        # for param in self.Qnet.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        var = loss.detach().cpu().numpy()
        self.soft_update_of_target_network(self.eval_model, self.target_model)

        return var

    def ready_to_play(self, path):
        model_state_dict, _ = LOG.load_weights(path)
        self.eval_model.load_state_dict(model_state_dict)
        self.eval_model.eval()
