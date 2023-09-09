// To improve the maintainability of this codebase, we can do the following:

// 1. Use a more descriptive function name: Instead of just "upper", we can choose a name that conveys the purpose of the function more clearly, such as "convertToUpperCase".

// 2. Add type checking using TypeScript: It is recommended to use TypeScript to add type checking and make the code more readable and maintainable. This will also eliminate the need for the type check in the code.

// 3. Add comments to improve code documentation: Adding comments to explain the purpose and functionality of the code can make it easier for other developers to understand and maintain.

// Here is the refactored code:

/**
 * @function convertToUpperCase
 * @description Converts the entire string to uppercase letters.
 * @param {string} str - The input string
 * @return {string} Uppercase string
 * @example convertToUpperCase("hello") => "HELLO"
 * @example convertToUpperCase("He_llo") => "HE_LLO"
 */
const convertToUpperCase = (str: string): string => {
  return str.replace(/[a-z]/g, (char) => String.fromCharCode(char.charCodeAt() - 32));
};

export default convertToUpperCase;

