def bitcount(n):
    # Initialize the count variable to 0
    count = 0
    
    # Loop through the bits of n until there are no more 1's left
    while n:
        # Set the least significant 1-bit to 0 and increment the count
        n &= n - 1
        count += 1
    
    # Return the count of 1-bits in n
    return count 