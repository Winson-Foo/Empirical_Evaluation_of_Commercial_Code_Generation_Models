// To improve the maintainability of this codebase, we can make the following changes:

// 1. Add comments to explain the purpose and logic of the code.
// 2. Make variable names more descriptive.
// 3. Use a loop to initialize the initial values of the Tribonacci sequence, instead of individually assigning them.
// 4. Break down the logic into smaller and more manageable functions.
// 5. Use clear and consistent indentation.

// Here's the refactored code:

/**
 * @function tribonacci
 * @description Calculates the Tribonacci number of a given input integer.
 * @param {Integer} n - The input integer.
 * @return {Integer} - The Tribonacci number of n.
 * @see [Tribonacci_Numbers](https://www.geeksforgeeks.org/tribonacci-numbers/)
 */
const tribonacci = (n) => {
  // creating array to store previous Tribonacci numbers
  const tribonacciNumbers = new Array(n + 1);

  // initialize the initial values of the Tribonacci sequence
  initializeTribonacci(tribonacciNumbers);

  // calculate Tribonacci numbers using a loop
  calculateTribonacci(tribonacciNumbers, n);

  // return the Tribonacci number of n
  return tribonacciNumbers[n];
};

/**
 * @function initializeTribonacci
 * @description Initializes the initial values of the Tribonacci sequence.
 * @param {Array} tribonacciNumbers - The array to store the Tribonacci numbers.
 */
const initializeTribonacci = (tribonacciNumbers) => {
  tribonacciNumbers[0] = 0;
  tribonacciNumbers[1] = 1;
  tribonacciNumbers[2] = 1;
};

/**
 * @function calculateTribonacci
 * @description Calculates the Tribonacci numbers using a loop.
 * @param {Array} tribonacciNumbers - The array to store the Tribonacci numbers.
 * @param {Integer} n - The input integer.
 */
const calculateTribonacci = (tribonacciNumbers, n) => {
  for (let i = 3; i <= n; i++) {
    tribonacciNumbers[i] = tribonacciNumbers[i - 1] + tribonacciNumbers[i - 2] + tribonacciNumbers[i - 3];
  }
};

export { tribonacci };

