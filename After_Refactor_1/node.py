class Node:
    def __init__(self, value=None):
        self.value = value
        self.successors = set()
        self.predecessors = set()

    def add_successor(self, node):
        self.successors.add(node)
        node.predecessors.add(self)

    def remove_successor(self, node):
        self.successors.remove(node)
        node.predecessors.remove(self)

    def get_successors(self):
        return self.successors.copy()

    def get_predecessors(self):
        return self.predecessors.copy()

    def has_successors(self):
        return bool(self.successors)

    def has_predecessors(self):
        return bool(self.predecessors)