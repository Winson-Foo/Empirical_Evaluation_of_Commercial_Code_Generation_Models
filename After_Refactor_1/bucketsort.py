def bucket_sort(arr: list[int], k: int) -> list[int]:
    """
    Sort a list of integers using the bucket sort algorithm.
    :param arr: List of integers to be sorted.
    :param k: Maximum value allowed for integers in arr.
    :return: Sorted list of integers.
    """
    counts = [0] * k

    # Count the frequency of each element in arr
    for x in arr:
        if x < 0 or x >= k:
            raise ValueError(f"Element {x} out of range [0, {k-1}]")
        counts[x] += 1

    # Construct the sorted list from the counts
    sorted_arr = [i for i, count in enumerate(counts) for _ in range(count)]
    return sorted_arr