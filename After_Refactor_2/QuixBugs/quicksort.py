def quicksort(arr):
    """Sorts an array in ascending order using the quicksort algorithm.

    Args:
        arr: A list of elements to be sorted.

    Returns:
        A sorted list of elements in ascending order.
    """

    # Check if array is empty
    if not arr:
        return []

    pivot = arr[0]

    # Partition array into elements smaller and greater than pivot
    lesser = quicksort([x for x in arr[1:] if x <= pivot])
    greater = quicksort([x for x in arr[1:] if x > pivot])

    # Combine and return sorted arrays
    return lesser + [pivot] + greater 