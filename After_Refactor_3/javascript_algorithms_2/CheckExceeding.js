// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add meaningful comments to describe the purpose and functionality of each code block and function.

// 2. Use descriptive variable names to enhance the code's readability and understanding.

// 3. Simplify the logic by removing unnecessary steps and variables.

// 4. Extract repetitive code into separate functions to improve code modularity and reusability.

// Here is the refactored code:

/**
 * @function checkExceeding
 * @description - Checks if there are exceeding words in a string. Exceeding words are words where the gap between two adjacent characters is increasing.
 * @param {string} str - The input string to check for exceeding words.
 * @returns {boolean} - True if there are exceeding words, false otherwise.
 * @example - checkExceeding('delete') => true, ascii difference - [1, 7, 7, 15, 15] which is incremental
 * @example - checkExceeding('update') => false, ascii difference - [5, 12, 3, 19, 15] which is not incremental
 */
const checkExceeding = (str) => {
  if (typeof str !== 'string') {
    throw new TypeError('Argument is not a string');
  }

  const getUpperCaseChars = (str) => {
    return str.toUpperCase().replace(/[^A-Z]/g, '');
  };

  const getAdjacentDifferences = (chars) => {
    const differences = [];
    for (let i = 0; i < chars.length - 1; i++) {
      const char = chars[i];
      const adjacentChar = chars[i + 1];
      if (char !== adjacentChar) {
        const diff = Math.abs(char.charCodeAt() - adjacentChar.charCodeAt());
        differences.push(diff);
      }
    }
    return differences;
  };

  const hasIncreasingDifferences = (differences) => {
    for (let i = 0; i < differences.length - 1; i++) {
      const diff = differences[i];
      const nextDiff = differences[i + 1];
      if (diff > nextDiff) {
        return false;
      }
    }
    return true;
  };

  const upperCaseChars = getUpperCaseChars(str);
  const adjacentDifferences = getAdjacentDifferences(upperCaseChars);
  const hasExceedingWords = hasIncreasingDifferences(adjacentDifferences);

  return hasExceedingWords;
};

export { checkExceeding };


