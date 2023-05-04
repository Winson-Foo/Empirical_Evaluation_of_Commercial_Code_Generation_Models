import heapq

def k_heap_sort(array, k):
    """
    Sorts a given array using k-heap sort algorithm

    Arguments:
    array -- the array to be sorted
    k -- the number of elements to keep in the heap

    Returns:
    A sorted version of the input array
    """
    heap = array[:k]
    heapq.heapify(heap)

    for element in array[k:]:
        yield heapq.heappushpop(heap, element)

    while heap:
        yield heapq.heappop(heap)