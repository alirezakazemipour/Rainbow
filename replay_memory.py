import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha

    def add(self, *item):
        self.memory.append(Transition(*item))
        self.priorities.append(max(self.priorities) if self.priorities else 1.0)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)
        assert len(self.memory) <= self.capacity

    def sample(self, batch_size, beta=0.4):
        priors = np.asarray(self.priorities) ** self.alpha
        total_priors = priors.sum()
        probabilities = (priors / total_priors).astype(np.float32).reshape(-1)
        sample_indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        weights = (len(self.memory) * probabilities[sample_indices]) ** (-beta)
        weights /= weights.max()

        return [self.memory[idx] for idx in sample_indices], weights, sample_indices

    def __len__(self):
        return len(self.memory)

    def update_priorities(self, indices, priors):
        for idx, prior in zip(indices, priors):
            self.priorities[idx] = prior
