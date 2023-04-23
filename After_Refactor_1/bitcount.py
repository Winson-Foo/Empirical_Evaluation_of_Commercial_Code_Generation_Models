def count_set_bits(number):
    """
    Counts the number of set bits (i.e. 1's) in a given integer.
    """
    set_bits_count = 0
    while number:
        number &= number - 1 # clears the least significant bit
        set_bits_count += 1
    return set_bits_count