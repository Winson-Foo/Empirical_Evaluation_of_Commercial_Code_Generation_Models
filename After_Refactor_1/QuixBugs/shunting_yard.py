def shunting_yard(tokens):
    """
    This function converts a list of infix tokens into a list of postfix tokens
    using the shunting yard algorithm.
    """
    # Define operator precedence
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    # Hold the resulting postfix tokens
    postfix_tokens = []

    # Stack for operators
    operator_stack = []

    # Iterate through each token
    for token in tokens:
        # If the token is a number, add to the output
        if isinstance(token, int):
            postfix_tokens.append(token)
        else:
            # If the token is an operator, pop operators from the stack
            # and append them to the output if they have higher or equal precedence
            while operator_stack and precedence[token] <= precedence[operator_stack[-1]]:
                postfix_tokens.append(operator_stack.pop())
            # Add the operator to the stack
            operator_stack.append(token)

    # Add any remaining operators to the output
    while operator_stack:
        postfix_tokens.append(operator_stack.pop())

    return postfix_tokens