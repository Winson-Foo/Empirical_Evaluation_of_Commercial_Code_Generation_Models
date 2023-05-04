def k_heap_sort(arr: list, k: int) -> list:
    """
    Implements a modified heapsort algorithm that sorts first k elements in O(klogk) and the remaining (n-k) elements
    in O((n-k)logk) time complexity.
    
    :param arr: Input list to sort
    :param k: Number of elements to sort initially
    :return: Sorted list
    """
    # Create a heap of first k elements
    k_heap = arr[:k]
    heapq.heapify(k_heap)

    # Insert elements greater than the k-th element
    for element in arr[k:]:
        heapq.heappushpop(k_heap, element)

    # Yield the sorted elements
    sorted_arr = []
    while k_heap:
        sorted_arr.append(heapq.heappop(k_heap))
    
    return sorted_arr

