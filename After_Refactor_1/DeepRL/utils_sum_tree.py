import numpy


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.empty(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _update_parent(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._update_parent(parent, change)

    def _find_leaf(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._find_leaf(left, s)
        else:
            return self._find_leaf(right, s - self.tree[left])

    def add(self, p, data):
        idx = self.n_entries + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.n_entries] = data
        self.update(idx, p)

        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._update_parent(idx, change)

    def get(self, s):
        idx = self._find_leaf(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return idx, self.tree[idx], data_idx

    def total(self):
        return self.tree[0]