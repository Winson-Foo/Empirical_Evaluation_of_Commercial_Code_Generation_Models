def get_subsequences(start, end, length):
    '''
    Returns all subsequences of given length within the specified range
    '''
    if length == 0:
        return [[]]

    subsequences = []
    for i in range(start, end + 1 - length):
        rest_of_subseq = get_subsequences(i + 1, end, length - 1)
        for rest in rest_of_subseq:
            subsequences.append([i] + rest)

    return subsequences 