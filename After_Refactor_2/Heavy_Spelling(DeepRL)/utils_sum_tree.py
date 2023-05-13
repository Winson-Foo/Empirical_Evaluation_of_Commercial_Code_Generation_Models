import numpy

class SumTree:
    def __init__(self, capacity):
        """
        Initialize a SumTree object with given capacity.
        """
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.number_of_entries = 0
        self.pending_indices = []

    def _propagate(self, index, change):
        """
        Recursively update parent nodes in the tree based on the change in priority.
        """
        parent_index = (index - 1) // 2
        self.tree[parent_index] += change
        if parent_index != 0:
            self._propagate(parent_index, change)

    def _retrieve(self, index, s):
        """
        Traverse the tree from root to leaf nodes to find the sample with given priority.
        """
        left_index = 2 * index + 1
        right_index = left_index + 1

        if left_index >= len(self.tree):
            return index

        if s <= self.tree[left_index]:
            return self._retrieve(left_index, s)
        else:
            return self._retrieve(right_index, s - self.tree[left_index])

    def total(self):
        """
        Return the total priority score of all stored samples.
        """
        return self.tree[0]

    def add(self, priority, data):
        """
        Add a sample with given priority and data to the SumTree.
        """
        index = self.number_of_entries + self.capacity - 1
        self.pending_indices.append(index)

        self.data[self.number_of_entries] = data
        self.update(index, priority)

        self.number_of_entries = (self.number_of_entries + 1) % self.capacity

    def update(self, index, priority):
        """
        Update the priority score of a sample at given index and propagate the change to parent nodes.
        """
        if index not in self.pending_indices:
            return
        self.pending_indices.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, s):
        """
        Find the sample with given priority score and return its index, priority, and data.
        """
        index = self._retrieve(0, s)
        data_index = index - self.capacity + 1
        self.pending_indices.append(index)
        return (index, self.tree[index], self.data[data_index])