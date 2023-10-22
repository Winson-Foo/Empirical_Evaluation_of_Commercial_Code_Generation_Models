import numpy

class SumTree:
    """A binary tree data structure where the parent�s value is the sum of its children."""

    def __init__(self, capacity: int):
        """Initialize the SumTree object."""
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx: int, change: float) -> None:
        """Update the value of a node and propagate the change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the index of a leaf node with a given priority value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get the total priority value of the SumTree."""
        return self.tree[0]

    def add(self, priority: float, data: object) -> None:
        """Add a new entry to the SumTree with a given priority value and data."""
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
        """Update the priority value of an existing entry in the SumTree."""
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> tuple:
        """Get the index, priority value, and data of an entry in the SumTree with a given value."""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], self.data[data_idx])

# Define constants and variables
CAPACITY = 100
EXAMPLE_DATA = "example"
EXAMPLE_PRIORITY = 0.5

# Create a SumTree object and test its functions
tree = SumTree(CAPACITY)

tree.add(EXAMPLE_PRIORITY, EXAMPLE_DATA)
assert tree.total() == EXAMPLE_PRIORITY

idx, priority, data = tree.get(EXAMPLE_PRIORITY)
assert idx == CAPACITY - 1
assert priority == EXAMPLE_PRIORITY
assert data == EXAMPLE_DATA

tree.update(idx, 1.0)
assert tree.total() == 1.0

idx, priority, data = tree.get(1.0)
assert idx == CAPACITY - 1
assert priority == 1.0
assert data == EXAMPLE_DATA