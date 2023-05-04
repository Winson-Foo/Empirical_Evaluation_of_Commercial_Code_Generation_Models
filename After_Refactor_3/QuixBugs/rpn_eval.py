class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop()

    def is_empty(self):
        return len(self._items) == 0


def rpn_eval(tokens):
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y
    }
    stack = Stack()

    for token in tokens:
        if isinstance(token, float):
            stack.push(token)
        else:
            b = stack.pop()
            a = stack.pop()
            stack.push(operators[token](a, b))

    return stack.pop() 