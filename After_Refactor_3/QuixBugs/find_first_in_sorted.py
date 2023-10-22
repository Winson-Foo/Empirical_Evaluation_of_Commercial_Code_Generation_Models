def find_first_occurrence(arr, target):
    left_end = 0
    right_end = len(arr)

    while left_end + 1 <= right_end:
        mid = (left_end + right_end) // 2

        if target == arr[mid] and (mid == 0 or target != arr[mid - 1]):
            return mid

        elif target <= arr[mid]:
            right_end = mid

        else:
            left_end = mid + 1

    return -1 