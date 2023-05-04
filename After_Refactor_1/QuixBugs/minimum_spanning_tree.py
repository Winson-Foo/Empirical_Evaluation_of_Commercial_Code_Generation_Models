def minimum_spanning_tree(weight_by_edge):
    group_by_node = {}
    mst_edges = set()

    sorted_edges = sort_edges(weight_by_edge)

    for edge in sorted_edges:
        u, v = edge
        belongs_to_same_group = group_nodes(u, v, group_by_node)
        if not belongs_to_same_group:
            mst_edges.add(edge)

    return mst_edges


def sort_edges(weight_by_edge):
    return sorted(weight_by_edge, key=weight_by_edge.__getitem__)


def group_nodes(u, v, group_by_node):
    if group_by_node.setdefault(u, {u}) == group_by_node.setdefault(v, {v}):
        return True
    else:
        new_group = group_by_node[u].union(group_by_node[v])
        for node in new_group:
            group_by_node[node] = new_group
        return False