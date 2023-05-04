from collections import deque

def breadth_first_search(start: Any, goal: Any) -> bool:
    """
    Search for the goal node from the start node using breadth-first search (BFS).
    Input:
        start: starting node for the BFS.
        goal: goal node to reach using the BFS.
    Output:
        bool: returns True if the goal node is found, else False.
    """
    queue = deque()
    queue.append(start)

    nodes_seen = set()
    nodes_seen.add(start)

    while queue:
        node = queue.popleft()

        if node is goal:
            return True
        else:
            successors = [n for n in node.successors if n not in nodes_seen]
            queue.extend(successors)
            nodes_seen.update(successors)

    return False 