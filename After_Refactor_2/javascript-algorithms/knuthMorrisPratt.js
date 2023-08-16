// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add comments to provide code documentation and enhance readability.
// 2. Use more descriptive variable names to improve code readability.
// 3. Extract the pattern table building logic into a separate function to improve modularity.
// 4. Remove unnecessary comments and unused code.

// Here is the refactored code:

// ```
/**
 * Builds the pattern table for the Knuth-Morris-Pratt algorithm.
 *
 * @param {string} word - The word to build the pattern table for.
 * @returns {number[]} - The pattern table.
 */
function buildPatternTable(word) {
  const patternTable = [0];
  let prefixIndex = 0;
  let suffixIndex = 1;

  while (suffixIndex < word.length) {
    if (word[prefixIndex] === word[suffixIndex]) {
      patternTable[suffixIndex] = prefixIndex + 1;
      suffixIndex += 1;
      prefixIndex += 1;
    } else if (prefixIndex === 0) {
      patternTable[suffixIndex] = 0;
      suffixIndex += 1;
    } else {
      prefixIndex = patternTable[prefixIndex - 1];
    }
  }

  return patternTable;
}

/**
 * Performs the Knuth-Morris-Pratt algorithm to find the first occurrence of a word in a text.
 *
 * @param {string} text - The text to search in.
 * @param {string} word - The word to search for.
 * @returns {number} - The index of the first occurrence of the word in the text, or -1 if not found.
 */
export default function knuthMorrisPratt(text, word) {
  if (word.length === 0) {
    return 0;
  }

  let textIndex = 0;
  let wordIndex = 0;

  const patternTable = buildPatternTable(word);

  while (textIndex < text.length) {
    if (text[textIndex] === word[wordIndex]) {
      // We've found a match.
      if (wordIndex === word.length - 1) {
        return (textIndex - word.length) + 1;
      }
      wordIndex += 1;
      textIndex += 1;
    } else if (wordIndex > 0) {
      wordIndex = patternTable[wordIndex - 1];
    } else {
      textIndex += 1;
    }
  }

  return -1;
}
// ```

// By following these improvements, the code becomes more understandable and maintainable.

