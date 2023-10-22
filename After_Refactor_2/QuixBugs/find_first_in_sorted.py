def find_first_in_sorted(arr, x):
    """
    Find the index of the first occurrence of x in a sorted list arr.

    Args:
    - arr: A sorted list of integers.
    - x: An integer to search for.

    Return:
    - The index of the first occurrence of x in arr if x is in arr, else -1.
    """
    
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid

        elif x <= arr[mid]:
            right = mid - 1

        else:
            left = mid + 1

    return -1