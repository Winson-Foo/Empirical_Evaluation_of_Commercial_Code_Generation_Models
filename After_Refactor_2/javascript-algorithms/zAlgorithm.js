// To improve the maintainability of the codebase, we can start by refactoring the `buildZArray` function. Here's the refactored code:

// ```javascript
/**
 * @param {string} zString
 * @return {number[]}
 */
function buildZArray(zString) {
  const zArray = [];
  const strLength = zString.length;

  for (let i = 0; i < strLength; i++) {
    zArray[i] = 0;
  }

  let left = 0;
  let right = 0;

  for (let i = 1; i < strLength; i++) {
    if (i > right) {
      let j = 0;

      while (i + j < strLength && zString[i + j] === zString[j]) {
        j++;
      }

      zArray[i] = j;

      if (j > 0) {
        left = i;
        right = i + j - 1;
      }
    } else {
      const k = i - left;
      const remainder = right - i + 1;

      if (zArray[k] < remainder) {
        zArray[i] = zArray[k];
      } else {
        let j = remainder;

        while (i + j < strLength && zString[i + j] === zString[right + j]) {
          j++;
        }

        zArray[i] = right - i + j;

        left = i;
        right = i + j - 1;
      }
    }
  }

  return zArray;
}
// ```

// Next, let's refactor the `zAlgorithm` function:

// ```javascript
export default function zAlgorithm(text, word) {
  const wordPositions = [];
  
  const zString = `${word}${SEPARATOR}${text}`;
  const zArray = buildZArray(zString);
  
  for (let i = word.length + SEPARATOR.length; i < zArray.length; i++) {
    if (zArray[i] === word.length) {
      wordPositions.push(i - word.length - SEPARATOR.length);
    }
  }
  
  return wordPositions;
}
// ```

// By refactoring the code in this way, we have made it more readable and easier to understand, which will improve the maintainability of the codebase.

