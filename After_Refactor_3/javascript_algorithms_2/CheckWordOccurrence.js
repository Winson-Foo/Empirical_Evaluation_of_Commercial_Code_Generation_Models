// To improve the maintainability of the codebase, we can make the following refactors:

// 1. Use more descriptive names for variables and functions.
// 2. Separate the input validation to a separate function.
// 3. Use a separate function to count the word occurrences.
// 4. Add comments to describe the purpose of each section of the code.

// Here's the refactored code:

/**
 * @function countWordOccurrences
 * @description - This function counts the occurrences of each word in a sentence and returns a word occurrence object.
 * @param {string} sentence
 * @param {boolean} isCaseSensitive
 * @returns {Object}
 */
const countWordOccurrences = (sentence, isCaseSensitive = false) => {
  validateInput(sentence, isCaseSensitive);

  const modifiedSentence = isCaseSensitive ? sentence.toLowerCase() : sentence;

  return countOccurrences(modifiedSentence);
};

/**
 * @function validateInput
 * @description - This function validates the input parameters.
 * @param {string} sentence
 * @param {boolean} isCaseSensitive
 * @throws {TypeError} - If the sentence is not a string or isCaseSensitive is not a boolean.
 */
const validateInput = (sentence, isCaseSensitive) => {
  if (typeof sentence !== 'string') {
    throw new TypeError('The sentence should be a string.');
  }

  if (typeof isCaseSensitive !== 'boolean') {
    throw new TypeError('The isCaseSensitive parameter should be a boolean.');
  }
};

/**
 * @function countOccurrences
 * @description - This function counts the occurrences of each word in a sentence and returns a word occurrence object.
 * @param {string} sentence
 * @returns {Object}
 */
const countOccurrences = (sentence) => {
  const words = sentence.split(/\s+/);
  const occurrence = {};

  words.forEach((word) => {
    occurrence[word] = occurrence[word] + 1 || 1;
  });

  return occurrence;
};

export { countWordOccurrences };

