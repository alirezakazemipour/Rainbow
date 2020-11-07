import numpy as np


class MinSegmentTree:
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0  # Full binary tree
        self.capacity = capacity
        self.tree = list(np.full(2 * self.capacity, np.inf))

    def query(self, start_idx, end_idx, current_node, first_node, last_node):
        if start_idx == first_node and end_idx == last_node:  # If we're on the node that contains what we want.
            return self.tree[current_node]
        mid_node = (first_node + last_node) // 2
        if mid_node >= end_idx:  # If the range lays completely on the left child
            return self.query(start_idx, end_idx, 2 * current_node, first_node, mid_node)
        elif mid_node + 1 <= start_idx:  # If the range lays completely on the right child
            return self.query(start_idx, end_idx, 2 * current_node + 1, mid_node, last_node)
        else:  # If the range lays partially on the left & right children
            return min(self.query(start_idx, mid_node, 2 * current_node, first_node, mid_node),  # Left part
                       self.query(mid_node + 1, end_idx, 2 * current_node + 1, mid_node + 1, last_node))  # Right part

    def min(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.capacity
        elif end_idx < 0:
            end_idx += self.capacity
        end_idx -= 1
        return self.query(start_idx, end_idx, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, value):
        idx += self.capacity
        self.tree[idx] = value
        # propagate the change through the tree.
        idx //= 2
        while idx >= 1:
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        idx += self.capacity
        return self.tree[idx]


class SumSegmentTree:
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0  # Full binary tree
        self.capacity = capacity
        self.tree = list(np.full(2 * self.capacity, 0))

    def query(self, start_idx, end_idx, current_node, first_node, last_node):
        if start_idx == first_node and end_idx == last_node:  # If we're on the node that contains what we want.
            return self.tree[current_node]
        mid_node = (first_node + last_node) // 2
        if mid_node >= end_idx:  # If the range lays completely on the left child
            return self.query(start_idx, end_idx, 2 * current_node, first_node, mid_node)
        elif mid_node + 1 <= start_idx:  # If the range lays completely on the right child
            return self.query(start_idx, end_idx, 2 * current_node + 1, mid_node, last_node)
        else:  # If the range lays partially on the left & right children
            return self.query(start_idx, mid_node, 2 * current_node, first_node, mid_node) + \
                   self.query(mid_node + 1, end_idx, 2 * current_node + 1, mid_node + 1,
                              last_node)  # Left + Right parts

    def sum(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.capacity
        elif end_idx < 0:
            end_idx += self.capacity
        end_idx -= 1
        return self.query(start_idx, end_idx, 1, 0, self.capacity - 1)

    def find_node(self, prior):
        assert 0 <= prior <= self.sum() + 1e-5

        idx = 1  # root
        while idx < self.capacity:
            if self.tree[2 * idx] > prior:  # Left child.
                idx *= 2
            else:
                prior -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    def __setitem__(self, idx, value):
        idx += self.capacity
        self.tree[idx] = value
        # propagate the change through the tree.
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        idx += self.capacity
        return self.tree[idx]
