import numpy

class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_indices = set()

    def _propagate(self, index, change):
        parent_index = (index - 1) // 2
        self.tree[parent_index] += change
        if parent_index != 0:
            self._propagate(parent_index, change)

    def _retrieve(self, index, s):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        index = self.n_entries + self.capacity - 1
        self.pending_indices.add(index)

        self.data[self.n_entries] = data
        self.update(index, priority)

        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    def update(self, index, priority):
        if index not in self.pending_indices:
            return
        self.pending_indices.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, s):
        index = self._retrieve(0, s)
        data_index = index - self.capacity + 1
        self.pending_indices.add(index)
        return (index, self.tree[index], data_index)