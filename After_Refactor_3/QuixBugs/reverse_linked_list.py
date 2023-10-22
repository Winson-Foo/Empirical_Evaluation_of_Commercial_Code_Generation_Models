def reverse_linked_list(node):
    """
    Reverses a linked list by changing the direction of the successor pointers.

    Parameters:
    node (Node): The head node of the linked list to be reversed.

    Returns:
    Node: The new head node of the reversed linked list.
    """
    prevnode = None
    while node:
        # Store the successor of the current node
        nextnode = node.successor
        # Point the successor of the current node to the previous node
        node.successor = prevnode
        # Move to the next node in the original linked list
        # and update the previous node variable
        prevnode, node = node, nextnode
    # Return the new head node of the reversed linked list
    return prevnode 