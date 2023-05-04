import heap_helpers

def shortest_path_length(length_by_edge, startnode, goalnode):
    """Finds the shortest path between two nodes in a directed graph.

    Inputs:
    - length_by_edge: A dictionary with every directed graph edge's length keyed
      by its corresponding ordered pair of nodes.
    - startnode: A node.
    - goalnode: A node.

    Preconditions:
    - All length values are positive.

    Output:
    - The length of the shortest path from startnode to goalnode in the input graph.
    """
    unvisited_nodes = []
    heap_helpers.insert_or_update(unvisited_nodes, (0, startnode))
    visited_nodes = set()

    return _shortest_path_helper(length_by_edge, unvisited_nodes, visited_nodes, goalnode)


def _shortest_path_helper(length_by_edge, unvisited_nodes, visited_nodes, goalnode):
    """Helper function to find the shortest path between two nodes.

    Inputs:
    - length_by_edge: A dictionary with every directed graph edge's length keyed
      by its corresponding ordered pair of nodes.
    - unvisited_nodes: A list of (node, distance) pairs in a heap order.
    - visited_nodes: A set of nodes that have already been visited.
    - goalnode: A node.

    Output:
    - The length of the shortest path from startnode to goalnode in the input graph.
    """
    while unvisited_nodes:
        distance, node = heap_helpers.pop_smallest(unvisited_nodes)
        if node is goalnode:
            return distance

        visited_nodes.add(node)

        for nextnode in node.successors:
            if nextnode in visited_nodes:
                continue

            heap_helpers.insert_or_update(unvisited_nodes,
                                           (min(heap_helpers.get_value(unvisited_nodes, nextnode),
                                                distance + length_by_edge[node, nextnode]),
                                            nextnode))

    return float('inf')


from heapq import heappop, heappush


def get_value(node_heap, wanted_node):
    """Returns the value corresponding to a given node in a heap.

    Inputs:
    - node_heap: A heap containing (value, node) pairs.
    - wanted_node: A node to find in the heap.

    Output:
    - The value corresponding to the given node, or 0 if the node is not found.
    """
    for value, node in node_heap:
        if node == wanted_node:
            return value
    return 0


def insert_or_update(node_heap, value_node):
    """Inserts a (value, node) pair into a heap, or updates its value if the node already exists.

    Inputs:
    - node_heap: A heap containing (value, node) pairs.
    - value_node: A (value, node) pair to insert or update.

    Output:
    - None.
    """
    value, node = value_node
    for i, tpl in enumerate(node_heap):
        a, b = tpl
        if b == node:
            node_heap[i] = value_node  # heapq retains sorted property
            return

    heappush(node_heap, value_node)


def pop_smallest(node_heap):
    """Removes the (value, node) pair with the smallest value from a heap.

    Inputs:
    - node_heap: A heap containing (value, node) pairs.

    Output:
    - The (value, node) pair with the smallest value.
    """
    return heappop(node_heap) 