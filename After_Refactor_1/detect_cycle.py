# This function detects whether a linked list contains a cycle or not.
# It uses the hare and tortoise algorithm to traverse the linked list.
# If the hare (which moves twice as fast as the tortoise) catches up with
# the tortoise, then there must be a cycle in the linked list.

def detect_cycle(head):
    # Initialize both pointers to the head of the linked list.
    slow = fast = head
    
    while True:
        # If either pointer reaches the end of the linked list, return False.
        if fast is None or fast.next is None:
            return False
        
        # Move the pointers forward.
        slow = slow.next
        fast = fast.next.next
        
        # If the pointers meet, return True (there is a cycle in the linked list).
        if slow is fast:
            return True
