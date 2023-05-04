def next_palindrome(digit_list):
    """
    Returns the next palindrome given a list of digits in the current number.

    digit_list: list[int] - A list of digits in the current number.
    """

    # Calculate the indices of the middle digits based on the length of the digit_list.
    middle_high = len(digit_list) // 2
    middle_low = (len(digit_list) - 1) // 2

    # Loop over the digit_list and increment digits until we reach a palindrome. 
    while middle_high < len(digit_list) and middle_low >= 0:
        if digit_list[middle_high] == 9:
            # If the middle digits are both 9, set them both to 0. 
            digit_list[middle_high] = 0
            digit_list[middle_low] = 0
            middle_high += 1
            middle_low -= 1
        else:
            # Increment the middle high digit and potentially the middle low digit.
            digit_list[middle_high] += 1
            if middle_high != middle_low:
                digit_list[middle_low] += 1
            return digit_list
    
    # If we didn't return from the while loop, we need to add a digit to either end to create a palindrome. 
    return [1] + (len(digit_list) - 1) * [0] + [1] 