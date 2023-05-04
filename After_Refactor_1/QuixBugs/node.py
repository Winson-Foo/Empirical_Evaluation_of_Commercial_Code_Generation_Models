class Node:
    def __init__(self, value=None):
        self.value = value
        self.successor = None
        self.successors = []
        self.predecessors = []
        self.incoming_nodes = []
        self.outgoing_nodes = []

    def get_successor(self):
        return self.successor

    def get_successors(self):
        return self.successors

    def get_predecessors(self):
        return self.predecessors