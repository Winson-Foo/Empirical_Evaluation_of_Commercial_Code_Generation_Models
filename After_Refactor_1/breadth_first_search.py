from collections import deque as Queue

def bfs(start_node, goal_node):
    """
    Perform a breadth-first search from the start_node to the goal_node.
    Return True if goal_node is found, False otherwise.
    """

    # Initialize the frontier queue and nodes seen set.
    frontier = Queue()
    frontier.append(start_node)
    nodes_seen = set()
    nodes_seen.add(start_node)

    # Iterate over nodes in the frontier queue until goal_node is found or queue is empty.
    while frontier:
        current_node = frontier.popleft()

        if current_node is goal_node:
            return True
        else:
            # Add child nodes to the frontier queue and nodes seen set.
            for child_node in current_node.successors:
                if child_node not in nodes_seen:
                    frontier.append(child_node)
                    nodes_seen.add(child_node)

    # If goal_node not found, return False.
    return False
