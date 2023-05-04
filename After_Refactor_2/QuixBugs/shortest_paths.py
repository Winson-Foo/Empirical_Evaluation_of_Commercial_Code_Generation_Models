def shortest_paths(source: int, edge_weights: dict) -> dict:
    """
    Calculates the shortest paths from a given source node to all other nodes
    in a weighted directed graph using the Bellman-Ford algorithm.

    Args:
    source (int): the index of the source node
    edge_weights (dict): a dictionary mapping edges to their weights

    Returns:
    A dictionary mapping nodes to their shortest path distances from the source node
    """

    # initialize the shortest path distances to infinity for all nodes
    node_weights = {node: float('inf') for node in set(sum(edge_weights.keys(), ()))}

    # set the shortest path distance to the source node to 0
    node_weights[source] = 0

    for _ in range(len(node_weights) - 1):
        relax_edges(node_weights, edge_weights)

    return node_weights

def relax_edges(node_weights: dict, edge_weights: dict):
    """
    Relaxes all edges in a given graph, updating the shortest path distances
    of the nodes it connects if a shorter path is found.

    Args:
    node_weights (dict): a dictionary mapping nodes to their current shortest path distances
    edge_weights (dict): a dictionary mapping edges to their weights
    """
    for (u, v), weight in edge_weights.items():
        if node_weights[u] + weight < node_weights[v]:
            # if the path to node v through u is shorter, update its shortest path distance
            node_weights[v] = node_weights[u] + weight 