import heapq

def heap_sort_k_largest(iterable, k_largest):
    """
    Returns a generator that sorts the given iterable using a heap algorithm,
    and yields the k largest items first, followed by the remaining items in sorted order.

    :param iterable: The iterable to be sorted.
    :param k_largest: The number of largest items to be yielded first.
    :return: A sorted generator.
    """
    heap = iterable[:k_largest]
    heapq.heapify(heap)

    for item in iterable[k_largest:]:
        yield heapq.heappushpop(heap, item)

    while heap:
        yield heapq.heappop(heap)