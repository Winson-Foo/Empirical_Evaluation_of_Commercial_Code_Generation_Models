def binary_search(array, item):
    """
    This function searches an item in a sorted array using the binary search algorithm.
    :param array: A sorted array.
    :param item: The item to be searched within the array.
    :return: The index of the item if found, otherwise -1.
    """
    low = 0
    high = len(array) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = array[mid]

        if guess == item:
            return mid

        if guess > item:
            high = mid - 1

        else:
            low = mid + 1

    return -1 