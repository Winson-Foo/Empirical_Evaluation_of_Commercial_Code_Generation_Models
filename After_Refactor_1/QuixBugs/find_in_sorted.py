def find_in_sorted(arr, x):
    """
    This function searches for a specified element in a sorted list using binary search algorithm.
    :param arr: The sorted list to search in.
    :param x: The element to search for.
    :return: The index of the specified element in the list, or -1 if not found.
    """
    def binsearch(start, end):
        """
        This recursive function implements binary search algorithm to find the specified element in the list.
        :param start: The index of the first element to consider.
        :param end: The index of the last element to consider.
        :return: The index of the specified element in the list, or -1 if not found.
        """
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        else:
            return mid

    return binsearch(0, len(arr))

def binary_search(sorted_list, target):
    """
    This function searches for a specified element in a sorted list using binary search algorithm.
    :param sorted_list: The sorted list to search in.
    :param target: The element to search for.
    :return: The index of the specified element in the list, or -1 if not found.
    """
    def binsearch(start_index, end_index):
        """
        This recursive function implements binary search algorithm to find the specified element in the list.
        :param start_index: The index of the first element to consider.
        :param end_index: The index of the last element to consider.
        :return: The index of the specified element in the list, or -1 if not found.
        """
        if start_index == end_index:
            return -1
        mid_index = start_index + (end_index - start_index) // 2
        if target < sorted_list[mid_index]:
            return binsearch(start_index, mid_index)
        elif target > sorted_list[mid_index]:
            return binsearch(mid_index + 1, end_index)
        else:
            return mid_index

    return binsearch(0, len(sorted_list))