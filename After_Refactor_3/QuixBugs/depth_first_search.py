def depth_first_search(start_node, goal_node):
    """A function to perform depth-first search on a graph"""
    visited_nodes = set()

    def search_from(node):
        """A recursive function that performs DFS"""
        if node in visited_nodes:
            return False
        elif node is goal_node:
            return True
        else:
            visited_nodes.add(node)
            return any(search_from(next_node) for next_node in node.successors)

    return search_from(start_node) 