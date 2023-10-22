from heapq import *


def shortest_path_length(length_by_edge, startnode, goalnode):
    unvisited_nodes = []
    heappush(unvisited_nodes, (0, startnode))
    visited_nodes = set()

    while len(unvisited_nodes) > 0:
        distance, node = heappop(unvisited_nodes)
        if node is goalnode:
            return distance

        visited_nodes.add(node)
        update_unvisited_nodes(unvisited_nodes, visited_nodes, length_by_edge, node, distance)

    return float('inf')


def update_unvisited_nodes(unvisited_nodes, visited_nodes, length_by_edge, node, distance):
    for nextnode in node.successors:
        if nextnode in visited_nodes:
            continue

        update_heap(unvisited_nodes, distance, nextnode, length_by_edge)


def get_distance(node_heap, wanted_node):
    for dist, node in node_heap:
        if node == wanted_node:
            return dist
    return 0


def update_heap(node_heap, distance, nextnode, length_by_edge):
    new_distance = min(
        get_distance(node_heap, nextnode) or float('inf'),
        distance + length_by_edge[node, nextnode]
    )
    dist_node = (new_distance, nextnode)

    for i, tpl in enumerate(node_heap):
        a, b = tpl
        if b == nextnode:
            node_heap[i] = dist_node
            return None

    heappush(node_heap, dist_node)
    return None 