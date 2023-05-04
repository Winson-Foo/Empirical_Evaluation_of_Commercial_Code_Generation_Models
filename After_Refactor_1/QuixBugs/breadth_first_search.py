import queue

def breadth_first_search(startnode, goalnode):
    # Create a queue to store nodes
    q = queue.Queue()
    q.put(startnode)

    # Create a set to store visited nodes
    visited = set()
    visited.add(startnode)

    # Loop until all nodes are visited or goalnode is found
    while not q.empty():
        node = q.get()

        # If goalnode is found, return True
        if node == goalnode:
            return True
        else:
            # Add unvisited successors to the queue and set
            for successor in node.successors:
                if successor not in visited:
                    q.put(successor)
                    visited.add(successor)

    # If goalnode is not found, return False
    return False