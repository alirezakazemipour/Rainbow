from torch import nn, from_numpy
import torch
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from logger import LOG

import numpy as np
from torch.optim.lr_scheduler import StepLR


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

# region model
# region model
class DQN(nn.Module):
    def __init__(self,
                 n_actions,
                 n_res_block,
                 n_stack,
                 state_shape,
                 filters,
                 first_stride=2,
                 fc_layer_unit=None
                 ):

        self.fc_layer_unit = fc_layer_unit
        super(DQN, self).__init__()
        self.n_actions = n_actions
        width, height, channels = state_shape

        self.conv1 = nn.Conv2d(channels, filters[0], kernel_size=3, padding=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(filters[0])

        self.resnet_lst = []
        for stack in range(n_stack):
            for block in range(n_res_block):
                if stack == 0:
                    input_filter = filters[0]
                elif block == 0:
                    input_filter = filters[stack - 1]
                else:
                    input_filter = filters[stack]

                if stack != 0 and block == 0:
                    stride = first_stride
                else:
                    stride = 1

                res_name = f'resnet_block_{stack * n_res_block + block}'
                self.resnet_lst.append(res_name)
                res_block = ResNetLayer(input_filter, filters[stack], first_stride=stride)
                setattr(self, res_name, res_block)

        conv_w = width
        conv_h = height

        if first_stride > 1:
            for _ in range(n_stack - 1):
                conv_w = get_conv_out(conv_w, kernel_size=3, padding=1, stride=first_stride)
                conv_h = get_conv_out(conv_h, kernel_size=3, padding=1, stride=first_stride)

        first_in = conv_w * conv_h * filters[-1]
        if fc_layer_unit is not None:
            self.fcs = list()
            for i, units in enumerate(fc_layer_unit):
                if i == 0:
                    in_unit = first_in
                else:
                    in_unit = fc_layer_unit[i - 1]
                fc_name = f'fc_{i}'
                self.fcs.append(fc_name)
                fc = nn.Linear(in_unit, units)
                setattr(self, fc_name, fc)
            first_in = units
            initial_weights(self.modules(), linear_initialization=True)
        else:
            initial_weights(self.modules())
        self.adv_value = nn.Linear(first_in, n_actions)
        self.s_value = nn.Linear(first_in, 1)
        # self.s_a_value = nn.Linear(first_in, self.n_actions)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        for res_name in self.resnet_lst:
            res_layer = getattr(self, res_name)
            x = res_layer(x)
        x = x.view(x.size(0), -1)
        if self.fc_layer_unit is not None:
            for fc_name in self.fcs:
                linear = getattr(self, fc_name)
                x = linear(x)
                x = F.relu(x)
        # x = self.s_a_value(x)
        adv_value = self.adv_value(x)
        s_value = self.s_value(x)
        x = s_value + adv_value - adv_value.mean(1, keepdim=True)
        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, filters, first_stride=1, strides=1, kernel_size=3):
        super(ResNetLayer, self).__init__()

        self.first_stride = first_stride
        if first_stride > 1:
            self.transition_layer = nn.Conv2d(in_channels=in_channels,
                                              out_channels=filters,
                                              kernel_size=1,
                                              stride=first_stride,
                                              # padding=1,
                                              )

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=filters,
                               kernel_size=kernel_size,
                               stride=first_stride,
                               padding=1)

        self.batch_norm = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(in_channels=filters,
                               out_channels=filters,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=1)
        initial_weights(self.modules())

    def forward(self, x):
        if self.first_stride > 1:
            inputs = self.transition_layer(x)
        else:
            inputs = x
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = self.conv2(x)
        return F.relu(x + inputs)


# endregion


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

        self.eval_model = DQN(state_shape=self.state_shape,
                              n_actions=self.n_actions,
                              n_res_block=1,
                              n_stack=2,
                              filters=[16, 32],
                              fc_layer_unit=[256, 256]).to(self.device)
        self.target_model = DQN(state_shape=self.state_shape,
                              n_actions=self.n_actions,
                              n_res_block=1,
                              n_stack=2,
                              filters=[16, 32],
                              fc_layer_unit=[256, 256]).to(self.device)

        if not TRAIN_FROM_SCRATCH:
            # TODO
            # Load weights and other params
            pass

        self.loss_fn = nn.MSELoss()
        # self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()  # Sets batchnorm and droupout for evaluation not training
        # self.optimizer = RMSprop(self.eval_model.parameters(), lr=self.lr, alpha=alpha)
        self.optimizer = Adam(self.eval_model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=int(100e3), gamma=0.1)
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
        # torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 100)  # clip gradients to help stabilise training

        # for param in self.Qnet.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.scheduler.step()
        var = loss.detach().cpu().numpy()
        self.soft_update_of_target_network(self.eval_model, self.target_model)

        return var

    def ready_to_play(self, path):
        model_state_dict, _ = LOG.load_weights(path)
        self.eval_model.load_state_dict(model_state_dict)
        self.eval_model.eval()
