// Here is the refactored code with improved maintainability:

/**
 * Get the greatest common divisor of two numbers.
 * @param {number} x - The first number.
 * @param {number} y - The second number.
 * @returns {number} The greatest common divisor of x and y.
 */
const gcdOfTwoNumbers = (x, y) => {
  while (x !== 0) {
    const temp = x;
    x = y % x;
    y = temp;
  }
  return y;
};

/**
 * Calculate the Euler's Totient Function (Phi) of a number n.
 * @param {number} n - The number.
 * @returns {number} The count of numbers in {1, 2, 3, ..., n} that are relatively prime to n.
 */
const eulersTotientFunction = (n) => {
  let countOfRelativelyPrimeNumbers = 0;
  for (let iterator = 1; iterator <= n; iterator++) {
    if (gcdOfTwoNumbers(iterator, n) === 1) {
      countOfRelativelyPrimeNumbers++;
    }
  }
  return countOfRelativelyPrimeNumbers;
};

export { eulersTotientFunction };


