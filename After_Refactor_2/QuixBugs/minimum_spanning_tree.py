def minimum_spanning_tree(edge_weight_map):
    """
    Computes the minimum spanning tree of a graph given the edge weights.
    :param edge_weight_map: A dictionary mapping edges to their weights.
    :return: A set of edges forming the minimum spanning tree.
    """
    nodes_by_group = {}  # maps nodes to their groups
    mst_edges = set()

    # Sort edges in increasing order of weight
    sorted_edges = sorted(edge_weight_map, key=edge_weight_map.__getitem__)

    for edge in sorted_edges:
        node1, node2 = edge
        group1 = nodes_by_group.setdefault(node1, {node1})
        group2 = nodes_by_group.setdefault(node2, {node2})

        # Add edge to MST if it connects groups from different nodes
        if group1 != group2:
            mst_edges.add(edge)

            # Merge the groups of the connected nodes
            group1.update(group2)
            for node in group2:
                nodes_by_group[node] = group1

    return mst_edges 