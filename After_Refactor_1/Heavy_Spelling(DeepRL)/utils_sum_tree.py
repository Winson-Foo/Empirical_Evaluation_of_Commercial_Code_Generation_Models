import numpy

class SumTree:
    # Use class constants instead of hard-coded values
    ROOT_NODE_INDEX = 0
    LEFT_CHILD_INDEX = lambda x: 2 * x + 1
    RIGHT_CHILD_INDEX = lambda x: 2 * x + 2

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.empty(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate_change_up(self, idx, change):
        parent_index = (idx - 1) // 2
        self.tree[parent_index] += change
        if parent_index != SumTree.ROOT_NODE_INDEX:
            self._propagate_change_up(parent_index, change)

    def _retrieve_leaf_index(self, search_value, idx=SumTree.ROOT_NODE_INDEX):
        left_child_index = SumTree.LEFT_CHILD_INDEX(idx)
        right_child_index = SumTree.RIGHT_CHILD_INDEX(idx)

        # if idx is a leaf node
        if left_child_index >= len(self.tree):
            return idx

        # if search_value is in the left subtree
        if search_value <= self.tree[left_child_index]:
            return self._retrieve_leaf_index(search_value, left_child_index)
        else:
            return self._retrieve_leaf_index(search_value - self.tree[left_child_index], right_child_index)

    def total_priority(self):
        return self.tree[SumTree.ROOT_NODE_INDEX]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update_priority(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update_priority(self, idx, priority):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate_change_up(idx, change)

    def get_leaf(self, search_value):
        idx = self._retrieve_leaf_index(search_value)
        data_index = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], self.data[data_index])