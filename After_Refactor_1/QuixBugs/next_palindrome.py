def next_palindrome(digit_list):
    left = 0
    right = len(digit_list) - 1
    while left < right:
        if digit_list[left] < digit_list[right]:
            digit_list[left:right+1] = [int(d) for d in str(int(''.join(map(str, digit_list)))+1)]
            return digit_list
        elif digit_list[left] > digit_list[right]:
            digit_list[right] = digit_list[left]
        left += 1
        right -= 1
    digit_list[left:right+1] = [int(d) for d in str(int(''.join(map(str, digit_list)))+1)]
    return digit_list