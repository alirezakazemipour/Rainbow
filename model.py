from torch import nn
import torch
import torch.nn.functional as F


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

        self.fc = nn.Linear(linear_input_size, 512)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        self.mass_probs = nn.Linear(512, self.n_actions * self.n_atoms)
        nn.init.xavier_uniform_(self.mass_probs.weight)
        self.mass_probs.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        return F.softmax(self.mass_probs(x).view(-1, self.n_actions, self.n_atoms),
                         dim=-1)  # (Batch size, N_Actions, N_Atoms)

    def get_q_value(self, x):
        dist = self(x)
        q_values = (dist * self.support).sum(dim=-1)  # (Batch size, N_Actions)
        return q_values