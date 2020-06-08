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

        self.adv_fc = nn.Linear(linear_input_size, 512)
        self.value_fc = nn.Linear(linear_input_size, 512)
        self.adv = nn.Linear(512, self.n_actions * self.n_atoms)
        self.value = nn.Linear(512, self.n_atoms)

        nn.init.kaiming_normal_(self.adv_fc.weight)
        self.adv_fc.bias.detach().zero_()

        nn.init.kaiming_normal_(self.value_fc.weight)
        self.value_fc.bias.detach().zero_()

        nn.init.xavier_uniform_(self.adv.weight)
        self.adv.bias.detach().zero_()

        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.detach().zero_()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv_fc = F.relu(self.adv_fc(x))
        value_fc = F.relu(self.value_fc(x))
        adv = self.adv(adv_fc).view(-1, self.n_actions, self.n_atoms)
        value = self.value(value_fc).view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value
