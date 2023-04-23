def get_nodes_with_no_incoming_nodes(nodes):
    return [node for node in nodes if not node.incoming_nodes]

def find_next_node(ordered_nodes, node):
    for nextnode in node.outgoing_nodes:
        incoming_nodes_set = set(nextnode.incoming_nodes)
        if incoming_nodes_set.issubset(ordered_nodes) and nextnode not in ordered_nodes:
            return nextnode
    return None

def topological_ordering(nodes):
    ordered_nodes = get_nodes_with_no_incoming_nodes(nodes)

    while len(ordered_nodes) < len(nodes):
        node = find_next_node(set(ordered_nodes), nodes)
        if not node:
            raise ValueError("Graph has a cycle.")
        ordered_nodes.append(node)

    return ordered_nodes