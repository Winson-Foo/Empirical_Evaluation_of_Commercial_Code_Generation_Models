def quicksort(arr):
    """
    Sorts a list using the quicksort algorithm.
    """

    if not arr:
        return []

    # Choose a pivot element to partition the list
    pivot = arr[0]

    # Partition the list into elements smaller and greater than the pivot
    lesser = quicksort([x for x in arr[1:] if x <= pivot])
    greater = quicksort([x for x in arr[1:] if x > pivot])

    # Combine the partitions and the pivot into a sorted list
    return lesser + [pivot] + greater