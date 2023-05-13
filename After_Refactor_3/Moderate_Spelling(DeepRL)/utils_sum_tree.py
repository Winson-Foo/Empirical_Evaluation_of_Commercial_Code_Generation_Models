import numpy

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _update_subtree(self, idx: int, change: float):
        '''
        Recursively update the priority values in the tree from the given index
        up to the root node, accounting for the given change in priority.
        '''
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._update_subtree(parent, change)

    def _find_leaf_index(self, s: float) -> int:
        '''
        Recursively traverse the tree from the root node to a leaf node, based on
        the given sample value, and return the index of the leaf node.
        '''
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._find_leaf_index(left, s)
        else:
            return self._find_leaf_index(right, s - self.tree[left])

    def total_priority(self) -> float:
        '''
        Return the sum of all priorities in the tree.
        '''
        return self.tree[0]

    def add(self, priority: float, data: object) -> None:
        '''
        Add a new priority-value pair to the tree, inserting it into the first
        available slot and updating the priority values accordingly.
        '''
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        '''
        Update the priority value at the given index, and propagate the change
        up the tree to all ancestor nodes.
        '''
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._update_subtree(idx, change)

    def get(self, sample: float) -> tuple:
        '''
        Get the priority-value pair corresponding to the given sample value by
        finding the leaf node in the tree that corresponds to the sample and
        returning the associated data and priority values.
        '''
        idx = self._find_leaf_index(0, sample)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], self.data[data_idx])