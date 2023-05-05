import numpy

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.n_entries % self.capacity + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.n_entries % self.capacity] = data
        self.update(idx, p)

        self.n_entries += 1

    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], self.data[data_idx])