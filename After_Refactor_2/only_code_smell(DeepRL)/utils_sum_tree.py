import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0
        self.pending_indices = set()

    def _propagate(self, index, change):
        parent_index = (index - 1) // 2
        self.tree[parent_index] += change
        if parent_index != 0:
            self._propagate(parent_index, change)

    def _get_left_child_index(self, index):
        return 2 * index + 1

    def _retrieve_index_with_sample(self, index, sample_value):
        left_child_index = self._get_left_child_index(index)
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            return index
        if sample_value <= self.tree[left_child_index]:
            return self._retrieve_index_with_sample(left_child_index, sample_value)
        else:
            return self._retrieve_index_with_sample(right_child_index, sample_value - self.tree[left_child_index])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        write_index = self.num_entries + self.capacity - 1
        self.pending_indices.add(write_index)
        self.data[self.num_entries % self.capacity] = data
        self.update_priority(write_index, priority)
        self.num_entries += 1

    def update_priority(self, index, priority):
        if index not in self.pending_indices:
            return
        self.pending_indices.remove(index)
        priority_diff = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, priority_diff)

    def get(self, sample_value):
        index = self._retrieve_index_with_sample(0, sample_value)
        data_index = index - self.capacity + 1
        self.pending_indices.add(index)
        return (index, self.tree[index], data_index)