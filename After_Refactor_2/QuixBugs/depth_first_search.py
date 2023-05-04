def depth_first_search(start_node, goal_node, nodes_visited=None):
    if nodes_visited is None:
        nodes_visited = set()

    if start_node in nodes_visited:
        return False
    elif start_node is goal_node:
        return True
    else:
        nodes_visited.add(start_node)
        return any(depth_first_search(next_node, goal_node, nodes_visited) 
                   for next_node in start_node.successors)