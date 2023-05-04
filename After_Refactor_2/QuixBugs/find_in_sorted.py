from typing import List

def find_index_in_sorted_array(array: List[int], target_item: int) -> int:
    """
    Return the index of the target item in the sorted array.

    Args:
    - array: A sorted list of integers.
    - target_item: An integer to search for.

    Returns:
    - The index of the target item in the array, or -1 if not found.
    """
    start = 0
    end = len(array) - 1

    while start <= end:
        mid = (start + end) // 2

        if target_item == array[mid]:
            # Target found, return index.
            return mid

        elif target_item < array[mid]:
            # Target is in left half of array.
            end = mid - 1

        else:
            # Target is in right half of array.
            start = mid + 1

    # Target not found in array.
    return -1