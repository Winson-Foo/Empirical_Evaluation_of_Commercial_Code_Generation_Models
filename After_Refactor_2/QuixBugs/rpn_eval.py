class Stack:
    """A simple stack implementation."""

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

def evaluate_reverse_polish_notation(tokens):
    """
    Evaluates a list of tokens in reverse Polish notation.

    Args:
        tokens: A list of numbers and arithmetic symbols.

    Returns:
        The result of the arithmetic expression.
    """
    def op(symbol, a, b):
        """
        Performs the arithmetic operation corresponding to the given symbol.

        Args:
            symbol: A string representing an arithmetic symbol (+, -, *, /).
            a: The first number to be used in the operation.
            b: The second number to be used in the operation.

        Returns:
            The result of the arithmetic operation.
        """
        return {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }[symbol](a, b)

    stack = Stack()

    for token in tokens:
        if isinstance(token, float):
            stack.push(token)
        else:
            a = stack.pop()
            b = stack.pop()
            stack.push(
                op(token, a, b)
            )

    return stack.pop() 