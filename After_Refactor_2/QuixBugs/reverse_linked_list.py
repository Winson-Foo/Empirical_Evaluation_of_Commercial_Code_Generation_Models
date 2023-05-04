def reverse_linked_list(head):
    # initialize variables
    prev_node = None
    current_node = head
    
    while current_node is not None:
        # save the successor of the current node
        next_node = current_node.successor
        
        # reverse the successor pointer of the current node
        current_node.successor = prev_node
        
        # move to the next node
        prev_node, current_node = current_node, next_node
        
    return prev_node 