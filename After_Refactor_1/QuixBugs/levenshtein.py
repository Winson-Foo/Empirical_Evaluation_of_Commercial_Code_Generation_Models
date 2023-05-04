def levenshtein(source, target):
    # Switch from if-elif-else to if-return to reduce nesting
    if source == '' or target == '':
        return len(source) or len(target)

    if source[0] == target[0]:
        # Use variables to make the recursive calls more readable
        source_remainder = source[1:]
        target_remainder = target[1:]
        return levenshtein(source_remainder, target_remainder)

    # Move the recursive calls to separate lines
    insert_cost = levenshtein(source, target[1:])
    delete_cost = levenshtein(source[1:], target)
    replace_cost = levenshtein(source[1:], target[1:])
    return 1 + min(insert_cost, delete_cost, replace_cost)