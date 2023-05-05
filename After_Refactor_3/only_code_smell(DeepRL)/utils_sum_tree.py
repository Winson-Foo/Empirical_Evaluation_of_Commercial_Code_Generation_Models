import numpy

class SumTree:
    WRITE_START_IDX = 0
    ROOT_IDX = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = numpy.zeros(self.tree_size)
        self.data = numpy.zeros(capacity, dtype=object)
        self.num_entries = 0
        self.pending_indices = []

    def _propagate(self, node_idx, change):
        parent_idx = (node_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != SumTree.ROOT_IDX:
            self._propagate(parent_idx, change)

    def _retrieve(self, node_idx, s):
        left_child_idx = 2 * node_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):
            return node_idx

        if s <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, s)
        else:
            return self._retrieve(right_child_idx, s - self.tree[left_child_idx])

    def total(self):
        return self.tree[SumTree.ROOT_IDX]

    def add(self, priority, data):
        write_idx = SumTree.WRITE_START_IDX + self.num_entries
        tree_idx = write_idx + self.capacity - 1
        self.pending_indices.append(tree_idx)

        self.data[write_idx] = data
        self.update(tree_idx, priority)

        SumTree.WRITE_START_IDX += 1
        if SumTree.WRITE_START_IDX >= self.capacity:
            SumTree.WRITE_START_IDX = 0

        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, tree_idx, priority):
        if tree_idx not in self.pending_indices:
            return
        self.pending_indices.remove(tree_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        tree_idx = self._retrieve(SumTree.ROOT_IDX, s)
        data_idx = tree_idx - self.capacity + 1
        self.pending_indices.append(tree_idx)
        return (tree_idx, self.tree[tree_idx], data_idx)