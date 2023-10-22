import heapq

def k_heapsort(values, k):
    """
    Sort the given iterable using a heap with a maximum size of k.
    """
    heap = values[:k]
    heapq.heapify(heap)

    for value in values[k:]:
        yield heapq.heappushpop(heap, value)

    while heap:
        yield heapq.heappop(heap) 