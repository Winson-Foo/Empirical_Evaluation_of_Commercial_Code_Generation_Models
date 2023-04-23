def shunting_yard(tokens):
    # define operator precedence
    operator_precedence = {
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2
    }
    
    # initialize RPNTokens and operator stack
    rpn_tokens = []
    operator_stack = []
    
    # loop through each token
    for token in tokens:
        # if it's an operand, add it to RPNTokens
        if isinstance(token, int):
            rpn_tokens.append(token)
        # if it's an operator
        else:
            # check the precedence
            while operator_stack and operator_precedence[token] <= operator_precedence[operator_stack[-1]]:
                # pop off the operator at the top of the stack if its precedence is higher
                rpn_tokens.append(operator_stack.pop())
            # add the operator to the stack
            operator_stack.append(token)
    
    # pop off any remaining operators in the stack and add them to RPNTokens
    while operator_stack:
        rpn_tokens.append(operator_stack.pop())
    
    # return the RPNTokens
    return rpn_tokens