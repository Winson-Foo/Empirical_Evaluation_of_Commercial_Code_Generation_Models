from typing import Dict, List, Tuple
from heapq import heappush, heappop


def shortest_path_length(edges: Dict[Tuple[str, str], int], start_node: str, goal_node: str) -> int:
    """
    Uses Dijkstra's algorithm to find the shortest path between two nodes in a directed graph.

    Args:
        edges: A dictionary with every directed graph edge's length keyed by its corresponding ordered pair of nodes
        start_node: The starting node
        goal_node: The goal node

    Returns:
        The length of the shortest path from startnode to goalnode in the input graph, or float('inf') if there is no path.

    Preconditions:
        all(length > 0 for length in edges.values())
    """
    unvisited_nodes = []  # A list of (distance, node) pairs
    heappush(unvisited_nodes, (0, start_node))
    visited_nodes = set()

    while unvisited_nodes:
        distance, node = heappop(unvisited_nodes)
        if node == goal_node:
            return distance

        visited_nodes.add(node)

        for next_node in get_successors(node, edges):
            if next_node in visited_nodes:
                continue

            update_distance(unvisited_nodes, edges, distance, node, next_node)

    return float('inf')


def update_distance(node_heap: List[Tuple[int, str]], edges: Dict[Tuple[str, str], int], curr_distance: int,
                     curr_node: str, next_node: str) -> None:
    """
    Inserts or updates the distance of the next_node in the priority queue.

    Args:
        node_heap: A priority queue (heap) containing (distance, node) pairs
        edges: A dictionary with every directed graph edge's length keyed by its corresponding ordered pair of nodes
        curr_distance: The current shortest distance from start_node to curr_node
        curr_node: The current node being considered
        next_node: The successor node of curr_node

    Returns:
        None.
    """
    new_distance = curr_distance + edges[(curr_node, next_node)]
    distance_to_next = get_distance(node_heap, next_node)

    if new_distance < distance_to_next:
        node_heap.append((new_distance, next_node))
        node_heap.sort()
    elif not distance_to_next:
        heappush(node_heap, (new_distance, next_node))
    else:
        # The new distance is not an improvement, so no need to update
        pass


def get_distance(node_heap: List[Tuple[int, str]], wanted_node: str) -> int:
    """
    Finds the distance of the given node in the priority queue.

    Args:
        node_heap: A priority queue (heap) containing (distance, node) pairs
        wanted_node: The node whose distance is being sought

    Returns:
        The distance of the wanted_node in the priority queue, or 0 if it is not in the queue.
    """
    for dist, node in node_heap:
        if node == wanted_node:
            return dist
    return 0


def get_successors(node: str, edges: Dict[Tuple[str, str], int]) -> List[str]:
    """
    Finds the successor nodes of the given node in the graph.

    Args:
        node: The node whose successors are being sought
        edges: A dictionary with every directed graph edge's length keyed by its corresponding ordered pair of nodes

    Returns:
        A list of the successor nodes of the given node.
    """
    successors = []
    for edge, length in edges.items():
        if edge[0] == node:
            successors.append(edge[1])
    return successors