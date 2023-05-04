def count_sort(input_list, max_val):
    # count the occurrences of each element
    counts = [0] * (max_val + 1)
    for num in input_list:
        counts[num] += 1

    # calculate running total of counts
    total = 0
    for i in range(max_val + 1):
        temp = counts[i]
        counts[i] = total
        total += temp

    # place elements in sorted order
    result = [0] * len(input_list)
    for num in input_list:
        result[counts[num]] = num
        counts[num] += 1

    return result