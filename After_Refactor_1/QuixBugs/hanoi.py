def hanoi(height, start_peg=1, end_peg=3):
    """
    This function solves the Tower of Hanoi puzzle recursively and returns the steps to solve it.

    Args:
        height (int): the height of the Tower of Hanoi puzzle
        start_peg (int): the peg where the puzzle starts (default is 1)
        end_peg (int): the peg where the puzzle ends (default is 3)

    Returns:
        list of tuples: the steps to solve the puzzle, where each tuple represents a move
                        and its format is (start_peg, end_peg)
    """
    steps = []

    if height > 0:
        helper_peg = get_helper_peg(start_peg, end_peg)
        
        steps.extend(hanoi(height - 1, start_peg, helper_peg))
        steps.append((start_peg, end_peg))
        steps.extend(hanoi(height - 1, helper_peg, end_peg))

    return steps


def get_helper_peg(start_peg, end_peg):
    """
    This function returns the peg that can be used as a helper peg to move the Tower of Hanoi puzzle.

    Args:
        start_peg (int): the peg where the puzzle starts
        end_peg (int): the peg where the puzzle ends

    Returns:
        int: the helper peg
    """
    helper_peg = ({1, 2, 3} - {start_peg} - {end_peg}).pop()
    return helper_peg