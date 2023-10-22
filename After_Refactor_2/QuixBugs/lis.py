from typing import List


def find_longest_increasing_subsequence(arr: List[int]) -> int:
    """
    Finds the length of the longest increasing subsequence in the given list.

    Parameters:
    arr (List[int]): A list of integers.

    Returns:
    int: The length of the longest increasing subsequence.
    """

    # Create a dictionary to track the ends of the longest increasing subsequence of each length
    ends_dict = {}
    longest = 0

    # Iterate over each value in the list
    for i, val in enumerate(arr):
        # Get the lengths of all prefixes that end with a value smaller than the current value
        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends_dict[j]] < val]

        # Get the length of the longest prefix that ends with a value smaller than the current value
        length = max(prefix_lengths) if prefix_lengths else 0

        # If the current value can be added to the end of the longest subsequence,
        # update the ends_dict and the length of the longest subsequence
        if length == longest or val < arr[ends_dict[length + 1]]:
            ends_dict[length + 1] = i
            longest = max(longest, length + 1)

    return longest