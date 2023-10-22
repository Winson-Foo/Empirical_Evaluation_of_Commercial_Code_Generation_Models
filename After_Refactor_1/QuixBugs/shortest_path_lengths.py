from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_shortest_paths(n: int, length_by_edge: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """
    Given a number of nodes, and the length of edges between them, calculates the shortest path
    between all pairs of nodes.

    :param n: The number of nodes
    :param length_by_edge: A dictionary with the length of edges between nodes.
    :return: A dictionary where the key is a tuple (i,j) representing the start and end node,
    and the value is the length of the shortest path starting at node i and ending at node j.
    """
    path_lengths = defaultdict(lambda: float('inf'))

    # Initialize the path_lengths dictionary with the length of individual edges.
    path_lengths.update({(i, i): 0 for i in range(n)})
    path_lengths.update(length_by_edge)

    # Run the Floyd-Warshall algorithm to calculate the shortest paths between all nodes.
    for k in range(n):
        for i in range(n):
            for j in range(n):
                path_lengths[i, j] = min(
                    path_lengths[i, j],
                    path_lengths[i, k] + path_lengths[k, j]
                )

    return path_lengths