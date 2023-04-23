from typing import Dict, List, Set, Tuple

def minimum_spanning_tree(weight_by_edge: Dict[Tuple[int, int], int]) -> Set[Tuple[int, int]]:
    """
    Given a dictionary where a key is a tuple of two vertices and the value
    is the weight of the edge between those vertices, returns the minimum
    spanning tree of the graph.

    :param weight_by_edge: A dictionary where the keys are tuples of two vertices
    and the values are the weight of the edge between those vertices.
    :return: A set of tuples, where each tuple is an edge in the minimum spanning tree.
    """

    def find_group(node: int, groups: Dict[int, Set[int]]) -> Set[int]:
        """
        Given a node and a dictionary of groups, returns the group that the node belongs to.

        :param node: An integer representing a node in a graph.
        :param groups: A dictionary where the keys are integers representing nodes
        and the values are sets of integers representing groups of nodes.
        :return: A set of integers representing the group that the node belongs to.
        """
        for group in groups.values():
            if node in group:
                return group

    # Create a dictionary where the keys are integers representing nodes
    # and the values are sets of integers representing groups of nodes.
    groups: Dict[int, Set[int]] = {}

    # Create an empty set to store the minimum spanning tree edges.
    mst_edges: Set[Tuple[int, int]] = set()

    # Sort the edges by weight in ascending order.
    edges: List[Tuple[int, int]] = sorted(weight_by_edge, key=weight_by_edge.__getitem__)

    # Iterate through the edges.
    for edge in edges:
        u, v = edge

        # If both nodes are not in the same group, add the edge to the minimum spanning tree.
        if find_group(u, groups) != find_group(v, groups):
            mst_edges.add(edge)

            # Merge the groups that u and v belong to.
            u_group = find_group(u, groups)
            v_group = find_group(v, groups)
            u_group.update(v_group)

            # Update the groups of all nodes in the merged group.
            for node in v_group:
                groups[node] = u_group

    return mst_edges