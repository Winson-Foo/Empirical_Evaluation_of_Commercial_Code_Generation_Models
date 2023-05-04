def hanoi(height, start=1, end=3, helper=2):
    """
    Solves the classic Towers of Hanoi problem using recursion.

    Args:
        height (int): The height of the stack of disks to solve for.
        start (int): The number of the starting peg (default 1).
        end (int): The number of the ending peg (default 3).
        helper (int): The number of the helper peg (default 2).

    Returns:
        A list of tuples representing the moves in the solution.
    """
    steps = []
    if height > 0:
        steps.extend(hanoi(height - 1, start, helper, end))
        steps.append((start, end))
        steps.extend(hanoi(height - 1, helper, end, start))

    return steps