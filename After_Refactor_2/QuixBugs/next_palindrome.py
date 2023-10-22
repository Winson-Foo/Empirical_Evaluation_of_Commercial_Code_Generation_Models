def next_palindrome(digit_list):
    high_mid = len(digit_list) // 2
    low_mid = (len(digit_list) - 1) // 2
    
    # Check for 9s in digit list
    if has_nines(digit_list):
        digit_list = zero_out_nines(digit_list, high_mid, low_mid)
        high_mid += 1
        low_mid -= 1
    
    # Add one to the left half of digit list
    digit_list = add_one(digit_list, high_mid, low_mid)
    
    # Handle case where adding one causes digit length to increase
    if len(digit_list) % 2 == 1:
        digit_list[low_mid + 1] = 0
    
    return digit_list

# Helper function to check if any nines in digit list
def has_nines(digit_list):
    return any(d == 9 for d in digit_list)

# Helper function to zero out nines in digit list
def zero_out_nines(digit_list, high_mid, low_mid):
    digit_list[high_mid] = digit_list[low_mid] = 0
    for i in range(high_mid+1, len(digit_list)):
        digit_list[i] = 0
    for i in range(low_mid-1, -1, -1):
        digit_list[i] = 0
    return digit_list

# Helper function to add one to left half of digit list
def add_one(digit_list, high_mid, low_mid):
    carry = 1
    i = low_mid
    j = high_mid if len(digit_list) % 2 == 0 else high_mid+1
    while i >= 0:
        d = digit_list[i] + carry
        carry = d // 10
        digit_list[i] = d % 10
        digit_list[j] = digit_list[i]
        i -= 1
        j += 1
    return digit_list 