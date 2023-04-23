def levenshtein(source, target):
    """
    This function calculates the Levenshtein distance between two strings
    :param source: Source string
    :param target: Target string
    :return: Levenshtein distance between source and target
    """

    # Base case: if either string is empty, return length of the other string
    if not source:
        return len(target)
    if not target:
        return len(source)

    # If the first character of both strings is the same, ignore it and move on to the next character
    if source[0] == target[0]:
        return levenshtein(source[1:], target[1:])

    # If the first character is different, consider all possible edits (insertion, deletion, substitution)
    # and choose the one that results in the smallest Levenshtein distance
    insert = levenshtein(source, target[1:])
    delete = levenshtein(source[1:], target)
    substitute = levenshtein(source[1:], target[1:])
    return 1 + min(insert, delete, substitute)