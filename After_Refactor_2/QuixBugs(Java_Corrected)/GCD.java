// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add comments to describe the purpose and functionality of the code.
// 2. Remove unnecessary imports.
// 3. Use more descriptive variable and method names.
// 4. Format the code properly for readability.

// Here is the refactored code:

// ```
package correct_java_programs;

/**
 * This class calculates the Greatest Common Divisor (GCD) of two numbers.
 */
public class GCD {

    /**
     * Calculates the GCD of two input numbers.
     *
     * @param num1 The first number.
     * @param num2 The second number.
     * @return The GCD of num1 and num2.
     */
    public static int calculateGCD(int num1, int num2) {
        if (num2 == 0) {
            return num1;
        } else {
            return calculateGCD(num2, num1 % num2);
        }
    }
}
// ```

// By making these changes, the code becomes more self-explanatory and easier to understand and maintain.

