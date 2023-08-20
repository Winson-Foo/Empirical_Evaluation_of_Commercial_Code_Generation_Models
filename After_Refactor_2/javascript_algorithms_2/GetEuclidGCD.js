// To improve the maintainability of the codebase, you can follow some best practices and refactor the code as follows:

// - Add comments to describe the purpose of the function and the logic being implemented.
// - Use meaningful variable names to improve code readability.
// - Separate the type validation into a separate function to improve code modularity and reusability.
// - Use strict equality (===) instead of loose equality (==) to avoid unexpected behavior.

// Here's the refactored code:

/**
 * getEuclidGCD - Euclidean algorithm to determine the GCD of two numbers.
 * @param {Number} a - Integer (may be negative).
 * @param {Number} b - Integer (may be negative).
 * @returns {Number} - Greatest Common Divisor gcd(a, b).
 */
export function getEuclidGCD(a, b) {
  validateInput(a, b);

  if (a === 0 && b === 0) {
    return undefined; // Infinitely many numbers divide 0.
  }

  let firstNumber = Math.abs(a);
  let secondNumber = Math.abs(b);

  while (secondNumber !== 0) {
    const remainder = firstNumber % secondNumber;
    firstNumber = secondNumber;
    secondNumber = remainder;
  }

  return firstNumber;
}

/**
 * validateInput - Validates if the arguments are numbers.
 * @param {Number} a - First number.
 * @param {Number} b - Second number.
 * @throws {TypeError} - Arguments must be numbers.
 */
function validateInput(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new TypeError('Arguments must be numbers');
  }
}

