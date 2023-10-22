def max_sublist_sum(arr):
    """
    Returns the maximum sublist sum for the given input array 
    """
    current_sum = 0
    max_sum = 0

    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum