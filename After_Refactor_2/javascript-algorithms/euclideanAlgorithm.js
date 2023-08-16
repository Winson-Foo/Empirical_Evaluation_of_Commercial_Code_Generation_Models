// To improve the maintainability of the codebase, you can follow these steps:

// 1. Add proper comments: Describe the purpose of the function and provide comments for each step in the code.

// 2. Use explanatory variable names: Replace generic variable names like 'a' and 'b' with more meaningful names like 'num1' and 'num2'.

// 3. Format the code consistently: Follow a consistent coding style, such as using spaces around operators, indenting code blocks properly, and using consistent line breaks.

// 4. Use parameter type annotations: Add type annotations to the function parameters to improve code readability and maintainability.

// 5. Separate the function into smaller functions: Break down the code into smaller functions to improve code readability and enable easier modification in the future.

// Here's the refactored code with the suggested improvements:

/**
 * Recursive version of Euclidean Algorithm of finding greatest common divisor (GCD).
 * @param {number} originalA - The first number.
 * @param {number} originalB - The second number.
 * @returns {number} - The greatest common divisor.
 */
export default function euclideanAlgorithm(originalA: number, originalB: number): number {
  // Make input numbers positive.
  const num1 = Math.abs(originalA);
  const num2 = Math.abs(originalB);

  return calculateGCD(num1, num2);
}

function calculateGCD(num1: number, num2: number): number {
  // To make the algorithm work faster instead of subtracting one number from
  // the other, we may use the modulo operation.
  return num2 === 0 ? num1 : calculateGCD(num2, num1 % num2);
}

