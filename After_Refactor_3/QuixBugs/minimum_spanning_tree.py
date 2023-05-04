def minimum_spanning_tree(weight_by_edge):
    # Convert the list of edges and their weights to a set for faster lookup
    edge_weights = {(u, v): w for (u, v), w in weight_by_edge}
    
    # Group each node into its own set initially
    groups = {node: {node} for edge in edge_weights for node in edge}

    # Sort the edges by their weights in increasing order
    edges_sorted = sorted(edge_weights.keys(), key=lambda e: edge_weights[e]) 
    
    # Use Kruskal's algorithm to find the minimum spanning tree
    mst_edges = set()
    for edge in edges_sorted:
        u, v = edge
        if groups[u] != groups[v]:
            mst_edges.add(edge)
            group_u, group_v = groups[u], groups[v]
            # Merge the groups containing u and v
            for node in group_v:
                group_u.add(node)
                groups[node] = group_u

    return mst_edges 