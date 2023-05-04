def detect_cycle(head_node):
    # Use fast and slow pointers to traverse the linked list
    slow_ptr = fast_ptr = head_node

    # Keep traversing the linked list until we reach the end or we detect a cycle
    while True:
        # If the fast pointer reaches the end, there is no cycle
        if fast_ptr is None or fast_ptr.successor is None:
            return False

        # Move slow and fast pointers ahead by one and two nodes, respectively
        slow_ptr = slow_ptr.successor
        fast_ptr = fast_ptr.successor.successor

        # If both pointers meet at the same node, it means we have detected a cycle
        if fast_ptr is slow_ptr:
            return True 