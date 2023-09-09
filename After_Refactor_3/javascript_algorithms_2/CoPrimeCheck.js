// To improve the maintainability of this codebase, we can make the following changes:

// 1. Extract the logic for calculating the greatest common divisor (GCD) into a separate function.
// 2. Use more descriptive variable names to improve readability.
// 3. Add error handling for invalid input types.
// 4. Use ES6 arrow function syntax for consistency.

// Here is the refactored code:

// Calculate the greatest common divisor (GCD) using Euclid's algorithm
const getEuclidGCD = (number1, number2) => {
  let smallerNumber = number1 > number2 ? number2 : number1;
  for (smallerNumber; smallerNumber >= 2; smallerNumber--) {
    if (number1 % smallerNumber === 0 && number2 % smallerNumber === 0) {
      return smallerNumber;
    }
  }
  return smallerNumber;
};

/**
 * Check if two numbers are coprime (relatively prime)
 * @param {Number} number1 The first number
 * @param {Number} number2 The second number
 * @returns {Boolean} True if the numbers are coprime, otherwise false
 */
const areNumbersCoprime = (number1, number2) => {
  // Check if the input is a number
  if (typeof number1 !== 'number' || typeof number2 !== 'number') {
    throw new TypeError('Arguments are not numbers.');
  }
  
  // Check if the numbers are coprime
  return getEuclidGCD(number1, number2) === 1;
};

export { areNumbersCoprime };

