from collections import defaultdict

def calculate_shortest_path_lengths(n, length_by_edge):
    """
    Calculates the shortest path lengths between all nodes in a graph.
    Args:
        n (int): The number of nodes in the graph.
        length_by_edge (dict): A dictionary containing the length of each edge in the graph.
    Returns:
        dict: A dictionary containing the shortest path lengths between all nodes in the graph.
    """
    length_by_path = defaultdict(lambda: float('inf'))
    length_by_path.update({(i, i): 0 for i in range(n)})
    length_by_path.update(length_by_edge)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                length_by_path[i, j] = min(
                    length_by_path[i, j],
                    length_by_path[i, k] + length_by_path[k, j]
                )
    
    return length_by_path 