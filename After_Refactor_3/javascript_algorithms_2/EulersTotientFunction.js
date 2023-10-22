// To improve the maintainability of the codebase, we can follow the following steps:

// 1. Add proper comments to explain the purpose and functionality of each function and variable.
// 2. Use meaningful and descriptive variable and function names to enhance understanding.
// 3. Break the code into smaller, reusable functions to improve readability and maintainability.
// 4. Remove unnecessary comments that restate the obvious.

// Here is the refactored code:

// ```javascript
/*
 * Author: sandyboypraper
 *
 * This function calculates Euler's Totient Function (phi).
 * Euler's Totient Function (phi) gives the count of numbers in {1,2,3,....,n} that are relatively
 * prime to n, i.e., the numbers whose GCD (Greatest Common Divisor) with n is 1.
 */

// Function to calculate the Greatest Common Divisor (GCD) of two numbers
const calculateGCD = (x, y) => {
  return x === 0 ? y : calculateGCD(y % x, x);
}

// Function to calculate Euler's Totient Function
const eulersTotientFunction = (n) => {
  let countOfRelativelyPrimeNumbers = 1;

  for (let iterator = 2; iterator <= n; iterator++) {
    if (calculateGCD(iterator, n) === 1) {
      countOfRelativelyPrimeNumbers++;
    }
  }

  return countOfRelativelyPrimeNumbers;
}

export { eulersTotientFunction };
// ```

// By following these steps, we have made the code more readable, maintainable, and easier to understand.

