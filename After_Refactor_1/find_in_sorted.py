def find_in_sorted(arr: List[int], x: int) -> int:
    """
    This method find element x in a sorted array using binary search.

    Args:
    -----
    arr (List[int]): a sorted array of integers
    x (int): the target element to be searched in the array

    Returns:
    --------
    mid (int): the position/index of the target element in the array. Returns -1 if not found.
    """

    def binsearch(start: int, end: int) -> int:
        if start == end:
            return -1

        mid = start + (end - start) // 2

        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        else:
            return mid

    return binsearch(0, len(arr))