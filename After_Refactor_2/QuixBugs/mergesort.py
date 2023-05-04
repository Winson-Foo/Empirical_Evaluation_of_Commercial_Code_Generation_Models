def mergesort(arr):
    # Check that the input array is not empty
    if not arr:
        raise ValueError("Input array cannot be empty")

    # Check that all elements in the array are of the same type
    element_type = type(arr[0])
    if any(type(element) != element_type for element in arr):
        raise TypeError("All elements in the array must be of the same type")

    def merge(left, right):
        # Merge two sorted subarrays into a sorted array
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:] or right[j:])
        return result

    # Check whether the length of the array is less than 2
    if len(arr) < 2:
        return arr
    else:
        # Divide the array into two halves and recursively sort each half
        middle = len(arr) // 2
        left = mergesort(arr[:middle])
        right = mergesort(arr[middle:])
        # Merge the sorted halves
        return merge(left, right)