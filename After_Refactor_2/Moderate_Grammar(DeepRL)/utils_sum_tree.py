# Here are some code refactors to improve maintainability:

import numpy

class SumTree:
    write_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.empty(capacity, dtype=object)
        self.num_entries = 0
        self.pending_updates = set()
    
    def _propagate(self, node_index):
        parent_index = (node_index - 1) // 2
        if parent_index < 0:
            return
        left_child_index = parent_index * 2 + 1
        right_child_index = left_child_index + 1
        left_child_value = self.tree[left_child_index]
        right_child_value = self.tree[right_child_index] if right_child_index < len(self.tree) else 0
        parent_value = left_child_value + right_child_value
        self.tree[parent_index] = parent_value
        self._propagate(parent_index)
    
    def _retrieve(self, node_index, sample_value):
        left_child_index = node_index * 2 + 1
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            return node_index
        left_child_value = self.tree[left_child_index]
        if sample_value <= left_child_value:
            return self._retrieve(left_child_index, sample_value)
        right_child_value = self.tree[right_child_index] if right_child_index < len(self.tree) else 0
        return self._retrieve(right_child_index, sample_value - left_child_value)
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        node_index = self.write_pointer + self.capacity - 1
        self.pending_updates.add(node_index)
        self.data[self.write_pointer] = data
        self.write_pointer = (self.write_pointer + 1) % self.capacity
        self.num_entries = min(self.num_entries + 1, self.capacity)
        self.update_priority(node_index, priority)
        
    def update_priority(self, node_index, priority):
        if node_index not in self.pending_updates:
            return
        self.pending_updates.remove(node_index)
        change = priority - self.tree[node_index]
        self.tree[node_index] = priority
        self._propagate(node_index)
        
    def get(self, sample_value):
        node_index = self._retrieve(0, sample_value)
        data_index = node_index - self.capacity + 1
        self.pending_updates.add(node_index)
        return (node_index, self.tree[node_index], self.data[data_index])