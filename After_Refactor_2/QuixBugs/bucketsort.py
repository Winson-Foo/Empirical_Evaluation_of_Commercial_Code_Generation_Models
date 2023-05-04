from typing import List

def bucketsort(input_list: List[int], num_buckets: int) -> List[int]:
    """
    Sorts a list of integers in ascending order using the bucket sort algorithm.

    Args:
        input_list: A list of integers to be sorted.
        num_buckets: The number of buckets to use in the sort.

    Returns:
        A new list of integers sorted in ascending order.
    """
    counts = [0] * num_buckets
    for x in input_list:
        counts[x] += 1

    sorted_arr = []
    for i, count in enumerate(counts):
        sorted_arr.extend([i] * count)

    return sorted_arr