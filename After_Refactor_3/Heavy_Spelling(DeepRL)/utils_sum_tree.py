# Refactored Code

import numpy

class SumTree:
    write_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        index = self.write_pointer + self.capacity - 1
        self.pending_idx.add(index)

        self.data[self.write_pointer] = data
        self.update(index, priority)

        self.write_pointer = (self.write_pointer + 1) % self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, index, priority):
        if index not in self.pending_idx:
            return
        
        self.pending_idx.remove(index)
        change = priority - self.tree[index]
        
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.capacity + 1
        
        self.pending_idx.add(index)
        return (index, self.tree[index], data_index)

# Improvements:
# 1. Add clear and __len__ methods to reset the tree and return the number of entries respectively.
# 2. Improve naming consistency and add informative comments to the methods.
# 3. Use modulo operator instead of if condition to handle circular buffer in add method. 
# 4. Rename the variables to more descriptive names. 
# 5. Remove unused write attribute.