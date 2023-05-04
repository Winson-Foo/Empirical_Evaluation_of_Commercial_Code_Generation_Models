class Node:
    def __init__(self, value=None, successor=None, successors=None, predecessors=None, incoming_nodes=None, outgoing_nodes=None):
        self.value = value
        self.successor_node = successor
        self.successor_list = successors if successors else []
        self.predecessor_list = predecessors if predecessors else []
        self.incoming_nodes = incoming_nodes if incoming_nodes else []
        self.outgoing_nodes = outgoing_nodes if outgoing_nodes else []

    def get_successor(self):
        return self.successor_node

    def get_successors(self):
        return self.successor_list

    def get_predecessors(self):
        return self.predecessor_list

    def add_successor(self, node):
        self.successor_list.append(node)

    def add_predecessor(self, node):
        self.predecessor_list.append(node)

    def add_incoming_node(self, node):
        self.incoming_nodes.append(node)

    def add_outgoing_node(self, node):
        self.outgoing_nodes.append(node) 