def kth(arr, k):
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]

    num_less = len(below)
    num_lessoreq = len(arr) - len(above)

    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k - num_lessoreq)
    else:
        return pivot


def find_k_smallest_element(arr, k):
    """Returns the kth smallest element in the given array.

    Args:
        arr (list): An array of integers.
        k (int): The position of the desired element in the sorted array.

    Returns:
        int: The kth smallest element in the array.
    """

    if not arr:
        return None

    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    above = [x for x in arr if x > pivot]

    num_less = len(below)
    num_equal = len(equal)
    num_greater = len(above)

    if k < num_less:
        return find_k_smallest_element(below, k)
    elif k < num_less + num_equal:
        return pivot
    else:
        return find_k_smallest_element(above, k - num_less - num_equal)