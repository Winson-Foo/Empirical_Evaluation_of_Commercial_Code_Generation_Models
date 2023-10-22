def reverse_linked_list(head_node):
    """
    Reverses a linked list.

    Args:
    - head_node: The head node of the linked list.

    Returns:
    - The new head node of the reversed linked list.
    """
    previous_node = None
    current_node = head_node

    while current_node:
        next_node = current_node.successor
        current_node.successor = previous_node
        previous_node, current_node = current_node, next_node

    return previous_node