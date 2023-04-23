def max_sublist_sum(arr):
    """
    Returns the maximum sum of any sublist within the given array

    Parameters:
    arr (list): the array of integers

    Returns:
    int: the maximum sum of any sublist within the array
    """

    # Initialize the maximum sum variables
    max_ending_here = 0
    max_so_far = 0

    # Iterate over each element in the array
    for num in arr:
        # Calculate the maximum sum of any sublist ending at the current element
        max_ending_here = max(num, max_ending_here + num)

        # Update the maximum sum of any sublist seen so far
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

def max_sublist_sum_range(arr, start, end):
    """
    Returns the maximum sum of any sublist within the given range of the array

    Parameters:
    arr (list): the array of integers
    start (int): the starting index of the range
    end (int): the ending index of the range

    Returns:
    int: the maximum sum of any sublist within the range of the array
    """

    # Initialize the maximum sum variables
    max_ending_here = 0
    max_so_far = 0

    # Iterate over each element in the range
    for num in arr[start:end+1]:
        # Calculate the maximum sum of any sublist ending at the current element
        max_ending_here = max(num, max_ending_here + num)

        # Update the maximum sum of any sublist seen so far
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far
