def max_sublist_sum(input_list):
    # initialize variables
    max_ending_here = 0
    max_so_far = 0

    # loop through input_list
    for x in input_list:
        # reset max_ending_here if it becomes negative
        max_ending_here = max(x, max_ending_here + x)
        
        # update max_so_far if necessary
        max_so_far = max(max_so_far, max_ending_here)

    # return the maximum sublist sum
    return max_so_far