from typing import List

def quicksort(arr: List[int]) -> List[int]:
    """
    Recursively sort a list of integers using the quicksort algorithm.
    :param arr: The list to be sorted.
    :return: The sorted list.
    """
    if not arr:
        return []

    pivot = arr[0]
    left = quicksort([x for x in arr[1:] if x <= pivot])
    right = quicksort([x for x in arr[1:] if x > pivot])
    return left + [pivot] + right 