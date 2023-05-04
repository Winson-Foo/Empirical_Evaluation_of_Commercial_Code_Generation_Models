def find_next_permutation(current_permutation):
    """
    Find the next lexicographically larger permutation of the given list of numbers.

    :param current_permutation: A list of numbers representing a permutation.
    :return: The next lexicographically larger permutation, or None if no such permutation exists.
    """

    # Find the largest index i such that perm[i] < perm[i+1]
    i = len(current_permutation) - 2
    while i >= 0 and current_permutation[i] >= current_permutation[i+1]:
        i -= 1

    if i == -1:
        # If i is the last element, there is no next permutation
        return None

    # Find the largest index j such that perm[i] < perm[j]
    j = len(current_permutation) - 1
    while j > i and current_permutation[j] <= current_permutation[i]:
        j -= 1

    # Swap perm[i] and perm[j]
    next_permutation = current_permutation.copy()
    next_permutation[i], next_permutation[j] = next_permutation[j], next_permutation[i]

    # Reverse the suffix starting at perm[i+1]
    next_permutation[i+1:] = reversed(next_permutation[i+1:])

    return next_permutation