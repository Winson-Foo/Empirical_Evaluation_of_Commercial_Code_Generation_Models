// To improve the maintainability of this codebase, you can make the following changes:

// 1. Add comments: One improvement that can be made is to add comments to explain the purpose of each step and clarify the code logic. This will make it easier for future developers to understand and maintain the code.

// 2. Use descriptive variable names: In the current code, the variable name `pat` is not descriptive enough to understand its purpose. Consider using a more meaningful variable name like `camelCasePattern`.

// 3. Use a more descriptive function name: The function name `checkCamelCase` is quite generic and does not accurately convey its purpose. Consider using a more descriptive function name like `isCamelCase`.

// 4. Modify the regular expression pattern: The regular expression pattern `^[a-z][A-Za-z]*$` is used to check for camel case. However, it does not account for numbers or special characters. Consider modifying the regular expression pattern to include these cases as well.

// Refactored code:

/**
 * isCamelCase method checks whether the given string is in camelCase or not.
 * @param {String} varName - The name of the variable to check.
 * @returns {Boolean} - Returns true if the string is in camelCase, else returns false.
 */
const isCamelCase = (varName) => {
  // Check if the input is a string or not.
  if (typeof varName !== 'string') {
    throw new TypeError('Argument is not a string.');
  }

  // Regular expression pattern to check for camelCase.
  const camelCasePattern = /^[a-zA-Z][a-zA-Z0-9]*$/;
  
  // Check if the string matches the camelCase pattern.
  return camelCasePattern.test(varName);
};

export { isCamelCase };

