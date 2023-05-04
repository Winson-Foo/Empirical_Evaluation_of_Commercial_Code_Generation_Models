from typing import List

class Node:
    def __init__(self, value=None):
        self.value = value
        self.successor_node = None
        self.successor_nodes: List[Node] = []
        self.predecessor_nodes: List[Node] = []

    def add_successor(self, node):
        self.successor_nodes.append(node)
        node.predecessor_nodes.append(self)

    def remove_successor(self, node):
        self.successor_nodes.remove(node)
        node.predecessor_nodes.remove(self)

    def add_predecessor(self, node):
        self.predecessor_nodes.append(node)
        node.successor_nodes.append(self)

    def remove_predecessor(self, node):
        self.predecessor_nodes.remove(node)
        node.successor_nodes.remove(self)

    def get_successor(self) -> 'Node':
        return self.successor_node

    def set_successor(self, node):
        self.successor_node = node

    def get_successors(self) -> List['Node']:
        return self.successor_nodes

    def get_predecessors(self) -> List['Node']:
        return self.predecessor_nodes 