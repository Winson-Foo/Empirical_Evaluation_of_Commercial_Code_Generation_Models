from typing import List

def find_first(arr: List[int], x: int) -> int:
    """
    Find the index of the first occurrence of x in a sorted array.

    Args:
        arr (List[int]): The sorted array to search.
        x (int): The value to search for.

    Returns:
        int: The index of the first occurrence of x in arr, or -1 if it is not present.
    """
    start = 0
    end = len(arr)

    while start < end:
        mid = (start + end) // 2

        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid

        elif x <= arr[mid]:
            end = mid

        else:
            start = mid + 1

    return -1