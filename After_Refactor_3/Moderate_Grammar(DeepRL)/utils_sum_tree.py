import numpy

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: object) -> None:
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
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> tuple[int, float, int]:
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], data_idx)