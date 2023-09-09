// To improve the maintainability of the codebase, we can make the following changes:
// - Add proper comments to explain the logic and purpose of each section of code.
// - Use descriptive variable and function names.
// - Split the code into smaller, reusable functions to improve readability and testability.
// - Provide error handling for incorrect input types.
// - Use ES6 arrow functions for conciseness.
// - Separate the code into different files/modules for better organization.

// Here is the refactored code:

// palindrome.js

/**
 * @function isPalindromeIntegerNumber
 * @param { Number } x
 * @returns {boolean} - input integer is palindrome or not
 *
 * time complexity : O(log_10(N))
 * space complexity : O(1)
 */
export function isPalindromeIntegerNumber(x) {
  if (!isInteger(x)) {
    throw new TypeError('Input must be an integer number');
  }

  if (x < 0) {
    return false;
  }

  const reversed = reverseNumber(x);

  return x === reversed;
}

/**
 * @function isInteger
 * @param { number } x
 * @returns { boolean } - whether the input is an integer or not
 */
function isInteger(x) {
  return typeof x === 'number' && Number.isInteger(x);
}

/**
 * @function reverseNumber
 * @param { number } x
 * @returns { number } - reversed version of the input number
 */
function reverseNumber(x) {
  let reversed = 0;
  let num = x;

  while (num > 0) {
    const lastDigit = num % 10;
    reversed = reversed * 10 + lastDigit;
    num = Math.floor(num / 10);
  }

  return reversed;
}

// // index.js

// import { isPalindromeIntegerNumber } from './palindrome.js';

// // Example usage
// console.log(isPalindromeIntegerNumber(12321)); // output: true
// console.log(isPalindromeIntegerNumber(12345)); // output: false
// console.log(isPalindromeIntegerNumber(-121)); // output: false

