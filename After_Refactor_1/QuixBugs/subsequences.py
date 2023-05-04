def generate_subsequences(start, end, length):
    if length == 0:
        return [[]]

    subsequences = []
    for i in range(start, end + 1 - length):
        subsequences.extend(
            [i] + rest for rest in generate_subsequences(i + 1, end, length - 1)
        )

    return subsequences