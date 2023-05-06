import numpy


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.num_entries = 0
        self.pending_updates = set()

    def propogate_updates(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self.propogate_updates(parent, change)

    def retrieve_sample(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve_sample(left, s)
        else:
            return self.retrieve_sample(right, s - self.tree[left])

    def get_total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.pending_updates.add(idx)

        self.data[self.write_idx] = data
        self.update_priority(idx, priority)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update_priority(self, idx, priority):
        if idx not in self.pending_updates:
            return

        self.pending_updates.remove(idx)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propogate_updates(idx, change)

    def get_priority_and_data(self, s):
        idx = self.retrieve_sample(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_updates.add(idx)
        return (self.tree[idx], self.data[data_idx])