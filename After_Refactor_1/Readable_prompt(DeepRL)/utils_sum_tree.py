import numpy

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    """

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        """
        Update the value of the parent nodes all the way to the root node.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        """
        Traverse the tree to find the sample on the leaf node.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self):
        """
        Get the sum of all values in the tree.
        """
        return self.tree[0]

    def add(self, priority, data):
        """
        Store the priority and the sample.
        """
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """
        Update the priority of the given index.
        """
        if idx not in self.pending_idx:
            return

        self.pending_idx.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value):
        """
        Get the priority and the index of the sample based on the value.
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return idx, self.tree[idx], data_idx