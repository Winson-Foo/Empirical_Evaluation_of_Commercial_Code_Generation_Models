import numpy

class SumTree:
    def __init__(self, capacity):
        """
        Initializes SumTree class.

        Parameters:
        capacity (int): Capacity of SumTree.
        """
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        """
        Propagates change from given index to the root node.

        Parameters:
        idx (int): Index to start propagation.
        change (float): Change value to be propagated.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Finds sample on leaf node.

        Parameters:
        idx (int): Index to start search.
        s (float): Sample to be searched.

        Returns:
        (int): Index of the leaf node containing given sample.
        """
        left = 2 * idx + 1
        right = left + 1
 
        if left >= len(self.tree):
            return idx
 
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        Returns total value of SumTree.

        Returns:
        (float): Total value of SumTree.
        """
        return self.tree[0]

    def add(self, p, data):
        """
        Stores priority and sample.

        Parameters:
        p (float): Priority value of the sample.
        data (object): Sample data to be stored.
        """
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """
        Updates the priority of given index.

        Parameters:
        idx (int): Index of the priority to be updated.
        p (float): New priority value.
        """
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Get priority and data of a sample closest to a given value.

        Parameters:
        s (float): Sample value to be searched for.

        Returns:
        (tuple): Tuple containing index, priority and data of the sample closest to s.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], self.data[dataIdx])