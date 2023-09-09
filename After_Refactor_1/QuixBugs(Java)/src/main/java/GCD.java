// To improve the maintainability of the codebase, you can follow these steps:

// 1. Improve code readability by using meaningful variable and method names.
// 2. Add comments to explain the purpose and functionality of the code.
// 3. Break long statements into multiple lines for better readability.
// 4. Use proper indentation to improve code structure.
// 5. Remove unnecessary imports and code.

// Here is the refactored code with the mentioned improvements:

package java_programs;

public class GCD {

    /**
     * Calculates the greatest common divisor (GCD) of two numbers.
     * 
     * @param a the first number
     * @param b the second number
     * @return the GCD of a and b
     */
    public static int calculateGCD(int a, int b) {
        if (b == 0) {
            return a;
        } else {
            return calculateGCD(a % b, b);
        }
    }
}

// Note that this is a basic refactoring. Depending on the specific requirements and context of your codebase, further improvements can be made.

