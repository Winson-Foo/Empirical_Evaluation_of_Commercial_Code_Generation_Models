// To improve the maintainability of the codebase, we can:

// 1. Split the code into smaller functions with clear and descriptive names.
// 2. Use constants and variable names that are more expressive.
// 3. Add comments to explain the purpose and logic of each section of code.
// 4. Implement error handling and validation for input parameters.

// Here's the refactored code:

// ```
const ZERO_OR_MORE_CHARS = '*';
const ANY_CHAR = '.';

/**
 * Checks if a string matches a given pattern using dynamic programming approach.
 *
 * @param {string} string
 * @param {string} pattern
 * @return {boolean}
 */
export default function regularExpressionMatching(string, pattern) {
  if (!string || !pattern) {
    throw new Error('Both string and pattern are required');
  }

  const matchMatrix = initializeMatchMatrix(string.length, pattern.length);
  fillFirstRow(matchMatrix, pattern);
  fillFirstColumn(matchMatrix, string);
  fillMatchMatrix(matchMatrix, string, pattern);

  return matchMatrix[string.length][pattern.length];
}

/**
 * Initializes the matchMatrix with null values.
 *
 * @param {number} rows
 * @param {number} columns
 * @return {Array.<Array.<boolean|null>>}
 */
function initializeMatchMatrix(rows, columns) {
  const matchMatrix = Array(rows + 1).fill(null).map(() => {
    return Array(columns + 1).fill(null);
  });

  matchMatrix[0][0] = true;

  return matchMatrix;
}

/**
 * Fills the first row of the matchMatrix.
 *
 * @param {Array.<Array.<boolean|null>>} matchMatrix
 * @param {string} pattern
 */
function fillFirstRow(matchMatrix, pattern) {
  for (let columnIndex = 1; columnIndex <= pattern.length; columnIndex += 1) {
    const patternIndex = columnIndex - 1;

    if (pattern[patternIndex] === ZERO_OR_MORE_CHARS) {
      matchMatrix[0][columnIndex] = matchMatrix[0][columnIndex - 2];
    } else {
      matchMatrix[0][columnIndex] = false;
    }
  }
}

/**
 * Fills the first column of the matchMatrix.
 *
 * @param {Array.<Array.<boolean|null>>} matchMatrix
 * @param {string} string
 */
function fillFirstColumn(matchMatrix, string) {
  for (let rowIndex = 1; rowIndex <= string.length; rowIndex += 1) {
    matchMatrix[rowIndex][0] = false;
  }
}

/**
 * Fills the remaining cells of the matchMatrix.
 *
 * @param {Array.<Array.<boolean|null>>} matchMatrix
 * @param {string} string
 * @param {string} pattern
 */
function fillMatchMatrix(matchMatrix, string, pattern) {
  for (let rowIndex = 1; rowIndex <= string.length; rowIndex += 1) {
    for (let columnIndex = 1; columnIndex <= pattern.length; columnIndex += 1) {
      const stringIndex = rowIndex - 1;
      const patternIndex = columnIndex - 1;

      if (pattern[patternIndex] === ZERO_OR_MORE_CHARS) {
        if (matchMatrix[rowIndex][columnIndex - 2] === true) {
          matchMatrix[rowIndex][columnIndex] = true;
        } else if (
          (
            pattern[patternIndex - 1] === string[stringIndex]
            || pattern[patternIndex - 1] === ANY_CHAR
          )
          && matchMatrix[rowIndex - 1][columnIndex] === true
        ) {
          matchMatrix[rowIndex][columnIndex] = true;
        } else {
          matchMatrix[rowIndex][columnIndex] = false;
        }
      } else if (
        pattern[patternIndex] === string[stringIndex]
        || pattern[patternIndex] === ANY_CHAR
      ) {
        matchMatrix[rowIndex][columnIndex] = matchMatrix[rowIndex - 1][columnIndex - 1];
      } else {
        matchMatrix[rowIndex][columnIndex] = false;
      }
    }
  }
}
// ```

// By splitting the code into smaller functions with clear responsibilities, adding comments, and using descriptive names for constants and variables, the code becomes easier to understand, maintain, and debug.

