def bucketsort(arr, k):
    """
    Sorts a list of integers using the bucket sort algorithm.
    :param arr: The list to be sorted
    :param k: The maximum value in the list
    :return: A sorted list of integers
    """
    if not isinstance(arr, list) or not isinstance(k, int):
        raise TypeError("Invalid input types")

    if k < 1:
        raise ValueError("Invalid bucket size")

    counts = [0] * (k + 1)  # Bucket list to keep track of counts
    for x in arr:
        if not isinstance(x, int):
            raise TypeError("Invalid input types")
        counts[x] += 1  # Increment the count of the current bucket

    sorted_arr = []
    for i in range(len(counts)):
        if counts[i] > 0:
            sorted_arr.extend([i] * counts[i])  # Add the number to the sorted array

    return sorted_arr 