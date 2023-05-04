from heapq import *


def shortest_path_length(length_by_edge, startnode, goalnode):
    # Use descriptive variable names
    unvisited_nodes = [] # MinHeap containing (distance, node) pairs
    heappush(unvisited_nodes, (0, startnode))
    visited_nodes = set()

    while unvisited_nodes:  # Use empty list as falsey value
        distance, node = heappop(unvisited_nodes)
        if node == goalnode: # Use == for comparison
            return distance

        visited_nodes.add(node)

        for nextnode, edge_length in length_by_edge[node].items():
            if nextnode in visited_nodes:
                continue

            # Use descriptive variable names
            next_distance = distance + edge_length
            insert_or_update(unvisited_nodes,
                             (next_distance, nextnode))

    return float('inf')


def insert_or_update(node_heap, dist_node):
    dist, node = dist_node
    for i, (a, b) in enumerate(node_heap):
        if b == node:
            # Use tuple packing and unpacking
            node_heap[i] = dist_node
            heapify(node_heap) # Re-heapify after modifying heap
            return None

    heappush(node_heap, dist_node)
    return None