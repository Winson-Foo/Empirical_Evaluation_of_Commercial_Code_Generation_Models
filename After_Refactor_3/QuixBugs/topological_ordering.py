def topological_ordering(nodes):
    # Use a set to keep track of nodes that have already been ordered
    ordered_nodes = set()

    # Create a dictionary to keep track of each node's incoming nodes
    incoming_nodes = {node: set() for node in nodes}

    # Populate the incoming nodes dictionary
    for node in nodes:
        for incoming_node in node.incoming_nodes:
            incoming_nodes[node].add(incoming_node)

    # Keep looping until all nodes have been ordered
    while len(ordered_nodes) < len(nodes):
        # Find the next node that can be ordered
        for node in nodes:
            if node not in ordered_nodes and incoming_nodes[node] <= ordered_nodes:
                ordered_nodes.add(node)
                break

    # Convert the set of ordered nodes back to a list
    ordered_nodes = list(ordered_nodes)

    return ordered_nodes 