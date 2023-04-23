def kth(arr, k):
    """
    Returns the kth smallest element in the array using quickselect algorithm.
    Assumes all elements in the array are distinct.

    Args:
    - arr (list[int]): An array of integers.
    - k (int): The index of the element to search for (starting from 0).

    Returns:
    - int: The kth smallest element in the array.
    """
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]

    num_less_than_pivot = len(below)
    num_less_or_equal_pivot = len(arr) - len(above)

    if k < num_less_than_pivot:
        # The kth element is in the below list
        return kth(below, k)
    elif k >= num_less_or_equal_pivot:
        # The kth element is in the above list
        return kth(above, k - num_less_or_equal_pivot)
    else:
        # The kth element is the pivot
        return pivot