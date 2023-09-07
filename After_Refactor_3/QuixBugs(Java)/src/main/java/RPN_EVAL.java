// Here is the refactored code with improved maintainability:

// ```java
package java_programs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;
import java.util.function.BinaryOperator;

public class RPN_EVAL {
    public static Double rpn_eval(ArrayList<Object> tokens) {
        Map<String, BinaryOperator<Double>> operators = createOperatorsMap();

        Stack<Double> stack = new Stack<>();

        for (Object token : tokens) {
            if (token instanceof Double) {
                stack.push((Double) token);
            } else {
                String oper = (String) token;
                Double a = stack.pop();
                Double b = stack.pop();
                Double result = calculate(operators.get(oper), a, b);
                stack.push(result);
            }
        }

        return stack.pop();
    }

    private static Map<String, BinaryOperator<Double>> createOperatorsMap() {
        Map<String, BinaryOperator<Double>> operators = new HashMap<>();
        operators.put("+", (a, b) -> a + b);
        operators.put("-", (a, b) -> a - b);
        operators.put("*", (a, b) -> a * b);
        operators.put("/", (a, b) -> a / b);
        return operators;
    }

    private static Double calculate(BinaryOperator<Double> operator, Double a, Double b) {
        return operator.apply(a, b);
    }
}
// ```

// In the refactored code:
// 1. The variable and method names are more informative and follow standard naming conventions.
// 2. The "op" map has been renamed to "operators".
// 3. The "stack" variable has been properly typed as Stack<Double> instead of just Stack.
// 4. The type check for Double has been changed to "instanceof" instead of using "Double.class.isInstance()".
// 5. The calculation logic has been moved to a separate method for improved readability.
// 6. The creation of the operators map has been moved to a separate method for better organization.
// 7. The imports have been sorted alphabetically for better readability.

