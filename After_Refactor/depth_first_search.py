def depth_first_search(start_node, goal_node):
    # Keep track of nodes that have been visited
    visited_nodes = set()

    # Recursive function to search for goal node
    def search_from(node):
        # If node has already been visited, stop searching
        if node in visited_nodes:
            return False
        # If we have found the goal node, stop searching
        elif node is goal_node:
            return True
        # Otherwise, mark the node as visited and search its successors
        else:
            visited_nodes.add(node)
            # Use a generator expression to search each successor node
            for next_node in node.successors:
                if search_from(next_node):
                    return True
            return False

    # Start the search from the start node
    return search_from(start_node)
