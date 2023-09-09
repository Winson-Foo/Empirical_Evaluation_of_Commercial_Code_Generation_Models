// To improve the maintainability of this codebase, we can make the following refactors:

// 1. Use meaningful variable names: Instead of "str" and "newStr", we can use more descriptive names like "inputString" and "cleanedString".

// 2. Separate the input validation: Move the input validation to a separate function to improve code readability and reusability.

// 3. Use built-in string functions: Instead of using the "at" method, we can directly access characters in a string using the bracket notation.

// 4. Break down the logic into smaller functions: Splitting the code into smaller functions will make it easier to understand and maintain.

// Here is the refactored code:

// ```javascript
/**
 * @function isAlphaNumericPalindrome
 * @description isAlphaNumericPalindrome should return true if the string has alphanumeric characters that are palindrome irrespective of special characters and the letter case.
 * @param {string} inputString the string to check
 * @returns {boolean}
 * @see [Palindrome](https://en.wikipedia.org/wiki/Palindrome)
 * @example
 * The function isAlphaNumericPalindrome() receives a string with varying formats
 * like "racecar", "RaceCar", and "race CAR"
 * The string can also have special characters
 * like "2A3*3a2", "2A3 3a2", and "2_A3*3#A2"
 *
 * But the catch is, we have to check only if the alphanumeric characters
 * are palindrome i.e remove spaces, symbols, punctuations etc
 * and the case of the characters doesn't matter
 */
const isAlphaNumericPalindrome = (inputString) => {
  if (!isValidString(inputString)) {
    throw new TypeError('Argument should be a string')
  }

  const cleanedString = cleanString(inputString)
  const midIndex = cleanedString.length >> 1 // x >> y = floor(x / 2^y)

  for (let i = 0; i < midIndex; i++) {
    if (cleanedString[i] !== cleanedString[~i]) { // ~n = -(n + 1)
      return false
    }
  }

  return true
}

const isValidString = (str) => {
  return typeof str === 'string'
}

const cleanString = (str) => {
  return str.replace(/[^a-z0-9]+/ig, '').toLowerCase()
}

export default isAlphaNumericPalindrome
// ```

// By implementing these refactors, the code becomes more readable, maintainable, and follows best practices.

