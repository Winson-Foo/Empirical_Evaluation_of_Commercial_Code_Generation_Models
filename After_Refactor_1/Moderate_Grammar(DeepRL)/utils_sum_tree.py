import numpy

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2*capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.num_entries = 0
        self.pending_indices = set()

    def _propagate_update_to_root(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate_update_to_root(parent, change)

    def _get_left_child_index(self, index):
        return 2*index + 1

    def _get_right_child_index(self, index):
        return self._get_left_child_index(index) + 1

    def _retrieve_leaf_node_index(self, index, sample):
        left_child_index = self._get_left_child_index(index)
        if left_child_index >= len(self.tree):
            return index
        if sample <= self.tree[left_child_index]:
            return self._retrieve_leaf_node_index(left_child_index, sample)
        else:
            return self._retrieve_leaf_node_index(self._get_right_child_index(index), sample - self.tree[left_child_index])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        index = self.num_entries + self.capacity - 1
        self.pending_indices.add(index)
        self.data[self.num_entries] = data
        self.update_priority(index, priority)
        self.num_entries += 1
        if self.num_entries >= self.capacity:
            self.num_entries = 0

    def update_priority(self, index, priority):
        if index not in self.pending_indices:
            return
        self.pending_indices.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate_update_to_root(index, change)

    def get(self, sample):
        leaf_index = self._retrieve_leaf_node_index(0, sample)
        data_index = leaf_index - self.capacity + 1
        self.pending_indices.add(leaf_index)
        return (leaf_index, self.tree[leaf_index], data_index)
