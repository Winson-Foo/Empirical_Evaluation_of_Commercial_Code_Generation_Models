def levenshtein(source, target):
    """
    Calculates the Levenshtein distance between two strings.
    """
    if source == '' or target == '':
        # Base case: one of the strings is empty, so the distance is the length of the other string
        return len(source) or len(target)

    elif source[0] == target[0]:
        # The first characters match, so we move on to the next characters
        return levenshtein(source[1:], target[1:])

    else:
        # The first characters don't match, so we try three different options and choose the one with the smallest distance
        insert = levenshtein(source, target[1:])
        delete = levenshtein(source[1:], target)
        replace = levenshtein(source[1:], target[1:])

        return 1 + min(insert, delete, replace) 