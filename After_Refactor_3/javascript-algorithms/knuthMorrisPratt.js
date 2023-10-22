// To improve the maintainability of this codebase, I suggest the following refactoring:

// 1. Split the `knuthMorrisPratt` function into smaller, reusable functions. This will make the code more modular and easier to understand and maintain.

// ```javascript
function knuthMorrisPratt(text, word) {
  if (word.length === 0) {
    return 0;
  }

  const patternTable = buildPatternTable(word);
  return search(text, word, patternTable);
}

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

function search(text, word, patternTable) {
  let textIndex = 0;
  let wordIndex = 0;

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
      // wordIndex = 0;
      textIndex += 1;
    }
  }

  return -1;
}
// ```

// 2. Remove the unnecessary comments that don't provide any additional information.


// The refactored code separates the main logic into smaller functions while keeping the overall structure intact, making it easier to understand and maintain.

