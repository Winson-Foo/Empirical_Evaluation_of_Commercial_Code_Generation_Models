// To improve the maintainability of the codebase, here are some suggestions:

// 1. Provide clear and concise comments: The existing code has comments, but they are not very clear and informative. It would be better to provide comments that explain the purpose and functionality of each section of the code.

// 2. Use descriptive variable names: The variable names in the code are not very descriptive. It would be more maintainable if we use meaningful names that indicate the purpose of the variable.

// 3. Split the code into smaller functions: The current function is doing multiple tasks - splitting the string into words and counting their occurrences. It would be better to split the code into smaller functions, each responsible for a single task. This would make the code more modular and easier to understand.

// 4. Add error handling: The current code throws a TypeError if the input parameters are of the wrong type. It would be better to handle these errors gracefully and provide meaningful error messages to the user.

// Here is the refactored code with these improvements:

// ```javascript
/**
 * Counts the occurrence of each word in a sentence and returns a word occurrence object.
 * @param {string} sentence - The input sentence.
 * @param {boolean} isCaseSensitive - Whether to consider the case while counting occurrences.
 * @returns {Object} - The word occurrence object.
 * @throws {TypeError} - If the input parameters are of the wrong type.
 */
const checkWordOccurrence = (sentence, isCaseSensitive = false) => {
  if (typeof sentence !== 'string') {
    throw new TypeError('The sentence should be a string')
  }

  if (typeof isCaseSensitive !== 'boolean') {
    throw new TypeError('The isCaseSensitive should be a boolean')
  }

  const modifiedSentence = isCaseSensitive ? sentence.toLowerCase() : sentence

  const words = splitSentenceIntoWords(modifiedSentence)
  const wordOccurrence = countWordOccurrence(words)

  return wordOccurrence
}

/**
 * Splits the sentence into an array of words.
 * @param {string} sentence - The input sentence.
 * @returns {string[]} - The array of words.
 */
const splitSentenceIntoWords = (sentence) => {
  return sentence.split(/\s+/)
}

/**
 * Counts the occurrence of each word in the array.
 * @param {string[]} words - The array of words.
 * @returns {Object} - The word occurrence object.
 */
const countWordOccurrence = (words) => {
  return words.reduce((occurrence, word) => {
    occurrence[word] = occurrence[word] + 1 || 1
    return occurrence
  }, {})
}

export { checkWordOccurrence }
// ```

// With these improvements, the code is more maintainable and easier to read and understand. It is divided into smaller functions with clear responsibilities, and error handling is included to provide better feedback to the users.

