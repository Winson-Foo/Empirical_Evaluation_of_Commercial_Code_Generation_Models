def get_longest_increasing_subsequence(input_list):
    """
    Returns the length of the longest increasing subsequence in the given list.

    Args:
        input_list (list): List of integers.

    Returns:
        int: Length of the longest increasing subsequence in the list.
    """
    ending_indices = {}
    max_length = 0

    for i, val in enumerate(input_list):
        # Find the lengths of all increasing subsequences that end before the current value.
        prefix_lengths = [j for j in range(1, max_length + 1) if input_list[ending_indices[j]] < val]
        
        # If there are no such subsequences, set the length to 0.
        length = max(prefix_lengths) if prefix_lengths else 0
        
        # If the current value can extend the longest subsequence, update the ending index and length.
        if length == max_length or val < input_list[ending_indices[length + 1]]:
            ending_indices[length + 1] = i
            max_length = max(max_length, length + 1)

    return max_length