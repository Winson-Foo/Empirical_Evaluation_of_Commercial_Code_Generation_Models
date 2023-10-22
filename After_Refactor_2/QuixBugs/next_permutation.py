def next_permutation(input_permutation):
    """
    Find the lexicographically next permutation of the given input_permutation.

    Args:
        input_permutation: A list representing the input permutation.

    Returns:
        A list representing the next permutation or None if input_permutation is already the last lexicographic permutation.
    """
    i = len(input_permutation) - 2
    while i >= 0 and input_permutation[i] >= input_permutation[i+1]:
        i -= 1
    if i == -1:
        return None
    j = len(input_permutation) - 1
    while input_permutation[j] <= input_permutation[i]:
        j -= 1
    next_permutation = list(input_permutation)
    next_permutation[i], next_permutation[j] = input_permutation[j], input_permutation[i]
    next_permutation[i+1:] = reversed(next_permutation[i+1:])
    return next_permutation 