def hanoi(height, start=1, end=3):
    """
    Return a list of steps to solve the Towers of Hanoi puzzle with the given
    height and starting and ending pegs.
    """
    steps = []
    
    if height > 0:
        # Determine the intermediate peg
        intermediate = ({1, 2, 3} - {start} - {end}).pop()
        
        # Move the top n-1 disks from the starting peg to the intermediate peg
        steps.extend(hanoi(height - 1, start, intermediate))
        
        # Move the nth disk from the starting peg to the ending peg
        steps.append((start, end))
        
        # Move the n-1 disks from the intermediate peg to the ending peg
        steps.extend(hanoi(height - 1, intermediate, end))

    return steps 