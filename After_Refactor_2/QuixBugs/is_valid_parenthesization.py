def is_valid_parenthesization(parens):
    def has_balanced_parentheses(parens):
        depth = 0
        for paren in parens:
            if paren == '(':
                depth += 1
            elif paren == ')':
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    return has_balanced_parentheses(parens) 