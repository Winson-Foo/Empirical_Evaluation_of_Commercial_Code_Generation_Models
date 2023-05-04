def get_nodes_with_no_incoming_edges(nodes):
    return [node for node in nodes if not node.incoming_nodes]

def get_nodes_satisfying_dependency(node, ordered_nodes):
    return [nextnode for nextnode in node.outgoing_nodes if set(ordered_nodes).issuperset(nextnode.incoming_nodes) and nextnode not in ordered_nodes]

def topological_ordering(nodes):
    ordered_nodes = get_nodes_with_no_incoming_edges(nodes)

    for node in ordered_nodes:
        ordered_nodes += get_nodes_satisfying_dependency(node, ordered_nodes)

    return ordered_nodes 