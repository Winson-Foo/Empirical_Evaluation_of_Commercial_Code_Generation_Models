import numpy

class SumTree:
    """A SumTree data structure that allows fast retrieval of the highest priority samples."""

    def __init__(self, capacity: int):
        """Initialize the SumTree with a given capacity."""
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_indices = set()

    def _propagate(self, index: int, change: float):
        """Update the priority values of parent nodes after a change in priority."""
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index: int, value: float) -> int:
        """Find the index of the leaf node that corresponds to a given value."""
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """Return the total priority values of all samples in the SumTree."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add a sample to the SumTree with a given priority value."""
        index = len(self.data) + self.n_entries - self.capacity
        self.pending_indices.add(index)

        self.data[self.n_entries % self.capacity] = data
        self.update(index, priority)

        self.n_entries += 1

    def update(self, index: int, priority: float):
        """Update the priority value of a sample in the SumTree."""
        if index not in self.pending_indices:
            return
        self.pending_indices.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, value: float) -> tuple[int, float, int]:
        """Get the sample with the highest priority that corresponds to a given value."""
        index = self._retrieve(0, value)
        data_index = index - len(self.data) + self.capacity - 1
        self.pending_indices.add(index)
        return (index, self.tree[index], data_index)