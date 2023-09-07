// To improve the maintainability of the codebase, you can make the following changes:

// 1. Add comments to explain the purpose and functionality of the code.
// 2. Use meaningful variable names that indicate their purpose.
// 3. Break down the logic into smaller, more manageable methods.
// 4. Use appropriate data structures and collections to enhance readability.

// Here is the refactored code:

// ```java
package java_programs;

public class GCD {

    /**
     * Calculates the greatest common divisor between two numbers.
     *
     * @param a the first number
     * @param b the second number
     * @return the greatest common divisor
     */
    public static int calculateGCD(int a, int b) {
        if (b == 0) {
            return a;
        } else {
            return calculateGCD(a % b, b);
        }
    }
}
// ```

// By making these changes, the codebase is more readable, easier to understand, and maintainable.

