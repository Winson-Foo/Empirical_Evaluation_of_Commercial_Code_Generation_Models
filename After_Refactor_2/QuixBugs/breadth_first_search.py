from collections import deque

def breadth_first_search(startnode, goalnode):
    queue = deque()
    queue.append(startnode)

    nodes_seen = set()
    nodes_seen.add(startnode)

    while queue:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            for successor in node.successors:
                if successor not in nodes_seen:
                    queue.append(successor)
                    nodes_seen.add(successor)

    return False