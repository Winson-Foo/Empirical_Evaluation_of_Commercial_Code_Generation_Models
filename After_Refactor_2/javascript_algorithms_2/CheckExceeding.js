// To improve the maintainability of the codebase, you can follow these steps:

// 1. Break down the code into smaller, more manageable functions to improve readability and reusability.

// 2. Use descriptive variable and function names to make the code more self-explanatory.

// 3. Add proper comments and documentation to explain the purpose and functionality of the code.

// 4. Use error handling techniques to handle any potential errors or exceptions gracefully.

// Here's the refactored code with the suggested improvements:

/**
 * @function checkExceeding
 * @description - Checks if the given string contains exceeding words
 * @param {string} str - The input string to check
 * @returns {boolean} - True if the string contains exceeding words, false otherwise
 */
const checkExceeding = (str) => {
  if (typeof str !== 'string') {
    throw new TypeError('Argument is not a string');
  }

  const cleanString = removeNonAlphabets(str);
  const upperChars = convertToUpperCase(cleanString);

  const adjacentDiffList = calculateAdjacentDifferences(upperChars);

  return isIncremental(adjacentDiffList);
};

/**
 * @function removeNonAlphabets
 * @description - Removes all non-alphabetic characters from the given string
 * @param {string} str - The input string to clean
 * @returns {string} - The cleaned string with only alphabets
 */
const removeNonAlphabets = (str) => {
  return str.replace(/[^A-Za-z]/g, '');
};

/**
 * @function convertToUpperCase
 * @description - Converts the given string to uppercase
 * @param {string} str - The input string to convert
 * @returns {string} - The uppercase string
 */
const convertToUpperCase = (str) => {
  return str.toUpperCase();
};

/**
 * @function calculateAdjacentDifferences
 * @description - Calculates the differences between adjacent characters in the given string
 * @param {string} str - The input string to calculate differences
 * @returns {number[]} - The list of differences between adjacent characters
 */
const calculateAdjacentDifferences = (str) => {
  const adjacentDiffList = [];

  for (let i = 0; i < str.length - 1; i++) {
    const char = str[i];
    const adjacentChar = str[i + 1];
    adjacentDiffList.push(Math.abs(char.charCodeAt() - adjacentChar.charCodeAt()));
  }

  return adjacentDiffList;
};

/**
 * @function isIncremental
 * @description - Checks if the given list of differences is incremental
 * @param {number[]} diffList - The input list of differences to check
 * @returns {boolean} - True if the differences are incremental, false otherwise
 */
const isIncremental = (diffList) => {
  for (let i = 0; i < diffList.length - 1; i++) {
    const diff1 = diffList[i];
    const diff2 = diffList[i + 1];
    if (diff1 > diff2) {
      return false;
    }
  }

  return true;
};

export { checkExceeding };

