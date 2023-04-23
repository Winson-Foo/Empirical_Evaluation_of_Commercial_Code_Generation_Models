def next_palindrome(digit_list):
    high_mid = len(digit_list) // 2
    low_mid = (len(digit_list) - 1) // 2

    while high_mid < len(digit_list) and low_mid >= 0:
        if digit_list[high_mid] == 9:
            digit_list[high_mid] = 0
            digit_list[low_mid] = 0
            high_mid += 1
            low_mid -= 1
        else:
            digit_list[high_mid] += 1
            if low_mid != high_mid:
                digit_list[low_mid] += 1
            
            # Reset all digits to the left of high_mid and right of low_mid
            for i in range(high_mid + 1, len(digit_list)):
                digit_list[i] = 0
            for i in range(0, low_mid):
                digit_list[i] = 0
                
            # Copy digits to the left of the midpoint to the right of the midpoint
            for i in range(low_mid, high_mid):
                digit_list[high_mid - (i - low_mid) - 1] = digit_list[i]
            
            return digit_list
        
    # If all digits are 9, increment the first and last digits and add a 1 in the middle
    return [1] + (len(digit_list) - 1) * [0] + [1]