import numpy


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    # Recursive function to propagate the changes to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Retrieve the leaf node that corresponds to a given sample s
    def _retrieve_leaf(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        # Leaf node found
        if left >= len(self.tree):
            return idx

        # Search in the left subtree
        if s <= self.tree[left]:
            return self._retrieve_leaf(left, s)

        # Search in the right subtree
        else:
            return self._retrieve_leaf(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # Add a new sample and its corresponding priority to the tree
    def add(self, p, data):
        idx = self.n_entries + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.n_entries] = data
        self.update(idx, p)

        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    # Update the priority of a sample by idx
    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # Retrieve the priority, the index of the corresponding data, and the index of the associated tree node
    def get(self, s):
        idx = self._retrieve_leaf(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return idx, self.tree[idx], data_idx