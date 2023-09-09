// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add comments to explain the purpose and functionality of the code.
// 2. Use meaningful variable names to improve code readability.
// 3. Use arrow function syntax for better code consistency.
// 4. Move the recursive function call to a separate line for better readability.

// Here's the refactored code:

// ```
/**
 * Recursive version of Euclidean Algorithm of finding greatest common divisor (GCD).
 *
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} - The greatest common divisor of the two numbers.
 */
const euclideanAlgorithm = (a, b) => {
  // Make input numbers positive.
  const num1 = Math.abs(a);
  const num2 = Math.abs(b);

  // To make the algorithm work faster, instead of subtracting one number from the other,
  // we may use the modulo operation.
  if (num2 === 0) {
    return num1;
  } else {
    return euclideanAlgorithm(num2, num1 % num2);
  }
};

export default euclideanAlgorithm;
// ```

// By following these refactoring suggestions, the codebase becomes more maintainable with clearer explanations, improved code readability, and consistent coding style.

