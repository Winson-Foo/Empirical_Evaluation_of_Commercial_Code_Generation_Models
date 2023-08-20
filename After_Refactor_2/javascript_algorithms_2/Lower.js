// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add type checking for the input parameter 'str'.
// 2. Extract the logic for converting characters to lowercase into a separate function for better readability.
// 3. Use more descriptive variable and function names.
// 4. Add comments to explain the purpose and behavior of the code.

// Here is the refactored code:

/**
 * @function convertToLowercase
 * @description Converts a single character to lowercase.
 * @param {String} char - The character to convert
 * @returns {String} Lowercase character
 */
const convertToLowercase = (char) => {
  // Convert the character's ASCII code to lowercase by adding 32
  return String.fromCharCode(char.charCodeAt() + 32);
}

/**
 * @function lower
 * @description Converts the entire string to lowercase letters.
 * @param {String} str - The input string
 * @returns {String} Lowercase string
 * @throws {TypeError} - If the input is not of type string
 */
const lower = (str) => {
  // Check if the input parameter is of type string
  if (typeof str !== 'string') {
    throw new TypeError('Invalid Input Type');
  }

  // Use regular expression to replace uppercase characters with lowercase characters using the helper function
  return str.replace(/[A-Z]/g, (char) => convertToLowercase(char));
}

export default lower

