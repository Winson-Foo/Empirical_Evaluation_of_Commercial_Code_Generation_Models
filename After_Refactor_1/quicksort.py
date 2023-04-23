def quicksort(arr):
    """A recursive implementation of quicksort that returns the sorted array."""
    if not arr:
        # Base case: empty array
        return []

    pivot = arr[0]
    lesser = quicksort([x for x in arr[1:] if x <= pivot])
    greater = quicksort([x for x in arr[1:] if x > pivot])
    # Combine the lesser, pivot, and greater arrays
    return lesser + [pivot] + greater
