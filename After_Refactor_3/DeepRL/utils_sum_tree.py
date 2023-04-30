import numpy

class SumTree:
    ROOT_INDEX = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.current_index = 0
        self.number_of_entries = 0
        self.pending_indexes = set()

    def _propagate_priority(self, index, change):
        """
        Recursively update the priority of parent nodes, starting from the given index.
        """
        parent_index = (index - 1) // 2
        self.tree[parent_index] += change
        if parent_index != self.ROOT_INDEX:
            self._propagate_priority(parent_index, change)

    def _get_leaf_index(self, index, priority):
        """
        Find the leaf index that corresponds to the given priority.
        """
        left_child_index = 2 * index + 1
        if left_child_index >= len(self.tree):
            return index

        if priority <= self.tree[left_child_index]:
            return self._get_leaf_index(left_child_index, priority)
        else:
            return self._get_leaf_index(left_child_index + 1, priority - self.tree[left_child_index])

    def get_total_priority(self):
        """
        Get the total priority of all samples in the sum tree.
        """
        return self.tree[self.ROOT_INDEX]

    def add(self, priority, data):
        """
        Add a new sample with given priority to the sum tree.
        """
        index = self.current_index + self.capacity - 1
        self.pending_indexes.add(index)

        self.data[self.current_index] = data
        self.update(index, priority)

        self.current_index += 1
        if self.current_index >= self.capacity:
            self.current_index = 0

        if self.number_of_entries < self.capacity:
            self.number_of_entries += 1

    def update(self, index, priority):
        """
        Update the priority of the given sample in the sum tree.
        """
        if index not in self.pending_indexes:
            return
        self.pending_indexes.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate_priority(index, change)

    def get_sample(self, priority):
        """
        Get the sample with the given priority, and its corresponding priority and data indexes.
        """
        index = self._get_leaf_index(self.ROOT_INDEX, priority)
        data_index = index - self.capacity + 1
        self.pending_indexes.add(index)
        return (index, self.tree[index], data_index)