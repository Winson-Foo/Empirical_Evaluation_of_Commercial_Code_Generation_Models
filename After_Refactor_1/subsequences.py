def get_all_subsequences(start_index, end_index, required_length):
    if required_length == 0:
        return [[]]

    subsequences_list = []
    for i in range(start_index, end_index + 1 - required_length):
        subsequences_list.extend(
            [i] + rest for rest in get_all_subsequences(i + 1, end_index, required_length - 1)
        )

    return subsequences_list