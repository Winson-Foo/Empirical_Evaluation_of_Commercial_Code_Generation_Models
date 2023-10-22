def bitcount(n):
    count_of_bits = 0
    while n:
        n &= n - 1
        count_of_bits += 1
    return count_of_bits