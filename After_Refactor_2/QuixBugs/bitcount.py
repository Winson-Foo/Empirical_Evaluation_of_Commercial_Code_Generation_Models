def count_bits(n):
    # This function counts the number of set bits in a binary representation of a number.
    # It uses Brian Kernighan's algorithm for counting bits.
    # Input: n - an integer
    # Output: the number of set bits in the binary representation of n
    
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count