// To improve the maintainability of this codebase, you can make the following changes:

// 1. Rename the function `checkWordOccurrence` to something more descriptive, like `getWordOccurrenceCount`. This provides better clarity on what the function does.

// 2. Break down the code into smaller, more manageable functions. This allows for better reusability and easier understanding of the code.

// 3. Use meaningful variable names to enhance code readability and maintainability.

// Here is the refactored code:

// ```javascript
/**
 * @function getWordOccurrenceCount
 * @description - This function counts the occurrences of each word in a sentence and returns a word occurrence object.
 * @param {string} sentence
 * @param {boolean} isCaseSensitive
 * @returns {Object}
 */
const getWordOccurrenceCount = (sentence, isCaseSensitive = false) => {
  if (typeof sentence !== 'string') {
    throw new TypeError('The first param should be a string');
  }

  if (typeof isCaseSensitive !== 'boolean') {
    throw new TypeError('The second param should be a boolean');
  }

  const modifiedSentence = isCaseSensitive ? sentence.toLowerCase() : sentence;

  const words = modifiedSentence.split(/\s+/);

  const wordOccurrence = words.reduce((occurrence, word) => {
    occurrence[word] = occurrence[word] + 1 || 1;
    return occurrence;
  }, {});

  return wordOccurrence;
};

export { getWordOccurrenceCount };
// ```

// Note: The refactored code mainly focuses on improving variable names and adding more descriptive comments to enhance code readability and maintainability. The functionality of the code remains the same as in the original code.

