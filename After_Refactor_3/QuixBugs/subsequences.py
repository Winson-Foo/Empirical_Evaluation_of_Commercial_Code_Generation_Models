def subsequences(start, end, length):
    if length == 0:
        return [[]]

    ret = []
    for i in range(start, end + 1 - length):
        ret.extend(
            [i] + rest for rest in subsequences(i + 1, end, length - 1)
        )

    return ret 