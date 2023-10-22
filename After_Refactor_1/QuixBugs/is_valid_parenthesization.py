def is_valid_parenthesization(parens):
    """
    Returns True if the parentheses are valid, False otherwise.
    """
    depth = 0
    for paren in parens:
        if paren == '(':
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    
    return depth == 0