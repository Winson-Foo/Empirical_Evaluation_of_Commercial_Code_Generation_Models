from typing import List

def shunting_yard(tokens: List[str]) -> List[str]:
    # Define operator precedence
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    # Convert infix notation to reverse polish notation
    def convert_to_rpn(tokens: List[str]) -> List[str]:
        rpntokens = []
        opstack = []
        for token in tokens:
            if isinstance(token, int):
                rpntokens.append(token)
            else:
                while opstack and precedence[token] <= precedence[opstack[-1]]:
                    rpntokens.append(opstack.pop())
                opstack.append(token)

        while opstack:
            rpntokens.append(opstack.pop())

        return rpntokens

    return convert_to_rpn(tokens) 