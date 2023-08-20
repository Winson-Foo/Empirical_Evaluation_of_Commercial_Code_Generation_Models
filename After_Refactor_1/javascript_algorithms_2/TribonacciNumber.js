// To improve the maintainability of this codebase, we can make a few changes:

// 1. Use clear variable names: Instead of using just "dp" as the name for the array to store previous tribonacci numbers, we can use a more descriptive name like "previousTribonacciNumbers".

// 2. Add comments to explain the logic: Although the code is fairly straightforward, adding comments will make it easier for other developers to understand the purpose and logic behind each step.

// 3. Extract the calculation logic into a separate function: By extracting the calculation logic into a separate function, we can improve the reusability of the code and make it easier to read and understand.

// 4. Use destructuring assignment for array initialization: Instead of individually setting each element of the array, we can use destructuring assignment to initialize the array with the first three tribonacci numbers.

// Here is the refactored code:

// ```javascript
/**
 * @function getTribonacci
 * @description Calculate the tribonacci of a given number.
 * @param {Number} n - The input number.
 * @returns {Number} The tribonacci of n.
 * @see [Tribonacci_Numbers](https://www.geeksforgeeks.org/tribonacci-numbers/)
 */
const getTribonacci = (n) => {
  const previousTribonacciNumbers = [0, 1, 1];

  for (let i = 3; i <= n; i++) {
    const tribonacciNumber = previousTribonacciNumbers[i - 1] + previousTribonacciNumbers[i - 2] + previousTribonacciNumbers[i - 3];
    previousTribonacciNumbers.push(tribonacciNumber);
  }

  return previousTribonacciNumbers[n];
};

export { getTribonacci };
// ```

// By following these improvements, we have made the codebase more readable, maintainable, and easier to understand for future development.

