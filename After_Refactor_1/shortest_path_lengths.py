from collections import defaultdict

def calculate_shortest_path_lengths(n: int, edges: dict) -> dict:
    """
    Calculates the shortest path between nodes in a graph using the Floyd-Warshall algorithm.
    :param n: number of nodes in the graph
    :param edges: dictionary with edges and their weights {(node1, node2): weight}
    :return: dictionary with shortest path lengths {(node1, node2): shortest_path_length}
    """
    shortest_path_lengths = defaultdict(lambda: float('inf'))
    shortest_path_lengths.update({(i, i): 0 for i in range(n)})
    shortest_path_lengths.update(edges)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                shortest_path_lengths[(i, j)] = min(
                    shortest_path_lengths[(i, j)],
                    shortest_path_lengths[(i, k)] + shortest_path_lengths[(k, j)]
                )

    return shortest_path_lengths
