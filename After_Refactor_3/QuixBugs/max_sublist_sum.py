def max_sublist_sum(arr):
    """
    Returns the maximum sum of a contiguous subarray within the given array.

    Parameters:
    ----------
    arr: list
        the input array to find the maximum sum of a contiguous subarray

    Returns:
    -------
    max_so_far: int
        the maximum sum of a contiguous subarray within the given array
    """
    max_ending_here = max_so_far = 0

    for num in arr:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far 