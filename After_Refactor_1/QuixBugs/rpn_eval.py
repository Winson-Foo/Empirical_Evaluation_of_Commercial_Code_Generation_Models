class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
    
def rpn_eval(tokens):
    def op(symbol, operand1, operand2):
        if symbol == '+':
            return operand1 + operand2
        elif symbol == '-':
            return operand1 - operand2
        elif symbol == '*':
            return operand1 * operand2
        elif symbol == '/':
            return operand1 / operand2
    
    stack = Stack()

    for token in tokens:
        if isinstance(token, float):
            stack.push(token)
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            result = op(token, operand1, operand2)
            stack.push(result)

    return stack.pop()