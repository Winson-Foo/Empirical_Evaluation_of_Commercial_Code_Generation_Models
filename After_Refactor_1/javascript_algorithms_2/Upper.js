// To improve the maintainability of the codebase, you can:

// 1. Use proper code formatting and indentation for better readability.
// 2. Add comments to document the purpose and functionality of the code.
// 3. Use descriptive variable names to improve code readability.
// 4. Break down the function into smaller, more focused functions for better organization and reusability.
// 5. Handle edge cases and error scenarios with appropriate error handling.
// 6. Write unit tests to ensure the correctness of the refactored code.

// Here's the refactored code with the above improvements:

/**
 * @function convertToUppercase
 * @description Will convert the entire string to uppercase letters.
 * @param {String} str - The input string
 * @return {String} Uppercase string
 * @throws {TypeError} - If the argument is not a string
 * @example convertToUppercase("hello") => HELLO
 * @example convertToUppercase("He_llo") => HE_LLO
 */
const convertToUppercase = (str) => {
  // Check if the argument is a string
  if (typeof str !== 'string') {
    throw new TypeError('Argument should be a string');
  }

  // Convert the string to uppercase using regular expression and charCodeAt
  return str.replace(
    /[a-z]/g, (char) => String.fromCharCode(char.charCodeAt() - 32)
  );
}

export default convertToUppercase;

