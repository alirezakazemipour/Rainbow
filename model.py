from torch import nn
import torch
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - kernel_size) // stride + 1


class Model(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(Model, self).__init__()
        width, height, channel = state_shape
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=8, stride=4), kernel_size=4, stride=2)

        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, 512)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        self.q_values = nn.Linear(512, self.n_actions)
        nn.init.xavier_uniform_(self.q_values.weight)
        self.q_values.bias.data.zero_()

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
        x = F.relu(self.fc(x))

        return self.q_values(x)