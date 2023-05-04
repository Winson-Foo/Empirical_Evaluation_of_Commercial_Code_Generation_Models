def mergesort(arr):
    
    def merge(left, right):
        result = []
        left_index, right_index = 0, 0
        while left_index < len(left) and right_index < len(right):
            if left[left_index] <= right[right_index]:
                result.append(left[left_index])
                left_index += 1
            else:
                result.append(right[right_index])
                right_index += 1
        result += left[left_index:]
        result += right[right_index:]
        return result
    
    if len(arr) < 2:
        return arr
    
    middle = len(arr) // 2
    left = mergesort(arr[:middle])
    right = mergesort(arr[middle:])
    return merge(left, right) 