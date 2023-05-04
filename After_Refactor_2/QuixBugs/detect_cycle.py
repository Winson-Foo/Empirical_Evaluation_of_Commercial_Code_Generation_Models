def detect_cycle(node):
    hare = tortoise = node

    while hare.successor is not None and hare.successor.successor is not None:
        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True

    return False