def detect_cycle(starting_node):
    """
    Determines if there is a cycle in a linked list starting from the given node.
    Returns True if a cycle is found, False otherwise.
    """
    hare = tortoise = starting_node

    while True:
        # Check if hare has reached the end of the linked list
        if hare is None or hare.next_node is None:
            return False

        # Move the tortoise and hare pointers
        tortoise = tortoise.next_node
        hare = hare.next_node.next_node

        # Check if there is a cycle
        if hare is tortoise:
            return True