from typing import List

def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merge two sorted arrays into a single sorted array.
    """
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:] or right[j:])
    return result

def mergesort(arr: List[int]) -> List[int]:
    """
    Sort an array using the mergesort algorithm.
    """
    if len(arr) < 2:
        return arr
    else:
        middle = len(arr) // 2
        left = mergesort(arr[:middle])
        right = mergesort(arr[middle:])
        return merge(left, right)