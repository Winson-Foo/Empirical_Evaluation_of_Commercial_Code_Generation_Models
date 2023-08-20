// To improve the maintainability of the codebase, you can follow these steps:

// 1. Use meaningful variable and function names: Renaming variables and functions with descriptive names can make the code easier to understand and maintain. For example, instead of using `dp`, you can use `tribonacciNumbers` to indicate the purpose of the array.

// 2. Add comments for better code comprehension: Adding comments to explain complex logic or algorithms can make the codebase more maintainable for future developers. Comments should be concise and clear.

// 3. Break down complex logic into smaller functions: If a function is handling a large amount of logic, it can become difficult to maintain. Consider breaking down the logic into smaller functions, each responsible for a specific task. This makes it easier to understand and update individual parts of the code.

// 4. Use constants for magic numbers: Instead of using hardcoded numbers like 0, 1, and 2, define them as constants with meaningful names. This makes it easier to understand the purpose of those numbers in the code.

// Here's the refactored code with all these improvements:

/**
 * @function calculateTribonacci
 * @description Calculates the tribonacci number for a given input.
 * @param {Integer} n - The input integer
 * @return {Integer} - The tribonacci of n.
 * @see [Tribonacci_Numbers](https://www.geeksforgeeks.org/tribonacci-numbers/)
 */
const calculateTribonacci = (n) => {
  const tribonacciNumbers = new Array(n + 1);
  const INITIAL_TRIBONACCI_NUMBERS = [0, 1, 1];
  
  for (let i = 0; i < INITIAL_TRIBONACCI_NUMBERS.length; i++) {
    tribonacciNumbers[i] = INITIAL_TRIBONACCI_NUMBERS[i];
  }
  
  for (let i = 3; i <= n; i++) {
    tribonacciNumbers[i] = tribonacciNumbers[i - 1] + tribonacciNumbers[i - 2] + tribonacciNumbers[i - 3];
  }
  
  return tribonacciNumbers[n];
};

export { calculateTribonacci };

