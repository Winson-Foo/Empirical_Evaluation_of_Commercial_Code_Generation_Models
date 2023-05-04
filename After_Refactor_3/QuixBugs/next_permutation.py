def next_permutation(numbers):
    """
    This function takes a list of numbers and returns the next lexicographically greater permutation of those numbers.
    """
    
    # Iterate backwards over the list
    for index in range(len(numbers) - 2, -1, -1):
        # Check if the current element is less than the next element
        if numbers[index] < numbers[index + 1]:
            # Iterate backwards over the list again
            for inner_index in range(len(numbers) - 1, index, -1):
                # Check if the element at the inner_index is greater than the element at the current index
                if numbers[inner_index] > numbers[index]:
                    # Swap the elements at the current and inner_index
                    next_permutation = list(numbers)
                    next_permutation[index], next_permutation[inner_index] = numbers[inner_index], numbers[index]
                    # Reverse the elements after the current index
                    next_permutation[index + 1:] = reversed(next_permutation[index + 1:])
                    # Return the next permutation
                    return next_permutation
                    
    # If no next permutation is found, return the list sorted in ascending order
    return sorted(numbers) 