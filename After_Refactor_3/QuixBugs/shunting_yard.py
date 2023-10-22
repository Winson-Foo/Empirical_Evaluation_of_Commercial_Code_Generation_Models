def convert_to_rpn(tokens):
    # Convert an infix expression to reverse polish notation (RPN)
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    rpn_tokens = []
    operator_stack = []

    for token in tokens:
        if isinstance(token, int):
            rpn_tokens.append(token)
        else:
            while operator_stack and precedence[token] <= precedence[operator_stack[-1]]:
                rpn_tokens.append(operator_stack.pop())
            operator_stack.append(token)

    while operator_stack:
        rpn_tokens.append(operator_stack.pop())

    return rpn_tokens 