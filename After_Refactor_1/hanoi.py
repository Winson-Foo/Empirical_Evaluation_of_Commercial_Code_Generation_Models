def hanoi(height, start=1, end=3):
    """Solves the Towers of Hanoi problem recursively and returns a list of steps."""
    steps = []

    if height == 0:
        # Base case
        return steps

    # Calculate the helper rod
    helper = ({1, 2, 3} - {start} - {end}).pop()

    # Move the top height - 1 discs to the helper rod
    steps.extend(hanoi(height - 1, start, helper))

    # Move the remaining disc from the start rod to the end rod
    steps.append((start, end))

    # Move the height - 1 discs from the helper rod to the end rod
    steps.extend(hanoi(height - 1, helper, end))

    return steps