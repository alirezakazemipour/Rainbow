from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - kernel_size) // stride + 1


class Model(nn.Module):
    def __init__(self, state_shape, n_actions, n_atoms, support):
        super(Model, self).__init__()
        width, height, channel = state_shape
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.n_atoms = n_atoms
        self.support = support

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=8, stride=4), kernel_size=4, stride=2)

        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64

        self.adv_fc = NoisyLayer(linear_input_size, 512)
        self.adv = NoisyLayer(512, self.n_actions * self.n_atoms)

        self.value_fc = NoisyLayer(linear_input_size, 512)
        self.value = NoisyLayer(512, self.n_atoms)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        adv_fc = F.relu(self.adv_fc(x))
        adv = self.adv(adv_fc).view(-1, self.n_actions, self.n_atoms)
        value_fc = F.relu(self.value_fc(x))
        value = self.value(value_fc).view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1).clamp(min=1e-3)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value

    def reset(self):
        self.adv_fc.reset_noise()
        self.adv.reset_noise()
        self.value_fc.reset_noise()
        self.value.reset_noise()


class NoisyLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NoisyLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.mu_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.sigma_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.n_outputs, self.n_inputs))

        self.mu_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.sigma_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.n_outputs))

        self.mu_w.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_w.data.fill_(0.5 / np.sqrt(self.n_inputs))

        self.mu_b.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_b.data.fill_(0.5 / np.sqrt(self.n_outputs))

        self.epsilon_i = 0
        self.epsilon_j = 0
        self.reset_noise()

    def forward(self, inputs):
        x = inputs
        weights = self.mu_w + self.sigma_w * self.weight_epsilon
        biases = self.mu_b + self.sigma_b * self.bias_epsilon
        x = F.linear(x, weights, biases)
        return x

    def f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        self.epsilon_i = self.f(torch.randn(self.n_inputs))
        self.epsilon_j = self.f(torch.randn(self.n_outputs))
        self.weight_epsilon.copy_(self.epsilon_j.ger(self.epsilon_i))
        self.bias_epsilon.copy_(self.epsilon_j)
