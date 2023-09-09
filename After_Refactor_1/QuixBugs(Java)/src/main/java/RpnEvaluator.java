// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add generic types to the ArrayList and Stack to ensure type safety.
// 2. Use meaningful variable names that reflect their purpose.
// 3. Extract the operator map initialization to a separate method for better code organization.
// 4. Use the diamond operator to simplify the initialization of the operator map.
// 5. Use a try-catch block to handle possible exceptions when popping from the stack.
// 6. Remove unnecessary type casting.

// Here's the refactored code:

package java_programs;

import java.util.*;
import java.util.function.BinaryOperator;

public class RpnEvaluator {
    public static Double rpn_eval(ArrayList<Object> tokens) {
        Map<String, BinaryOperator<Double>> operators = initializeOperators();

        Stack<Double> stack = new Stack<>();

        for (Object token : tokens) {
            if (token instanceof Double) {
                stack.push((Double) token);
            } else {
                String operator = (String) token;
                try {
                    Double b = stack.pop();
                    Double a = stack.pop();
                    BinaryOperator<Double> binOp = operators.get(operator);
                    Double result = binOp.apply(a, b);
                    stack.push(result);
                } catch (EmptyStackException e) {
                    System.out.println("Invalid RPN expression: Insufficient operands");
                    return null;
                }
            }
        }

        return stack.pop();
    }

    private static Map<String, BinaryOperator<Double>> initializeOperators() {
        Map<String, BinaryOperator<Double>> operators = new HashMap<>();
        operators.put("+", (a, b) -> a + b);
        operators.put("-", (a, b) -> a - b);
        operators.put("*", (a, b) -> a * b);
        operators.put("/", (a, b) -> a / b);
        return operators;
    }
}

// Note: The above code assumes that the tokens list contains valid RPN expressions and doesn't handle any possible exceptions related to unsupported operators or insufficient operands. Additional error handling can be added as per your specific requirements.

