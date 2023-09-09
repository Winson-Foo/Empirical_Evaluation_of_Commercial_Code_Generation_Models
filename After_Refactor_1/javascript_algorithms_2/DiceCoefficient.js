// To improve the maintainability of this codebase, we can consider the following refactored code:

/* The Sørensen–Dice coefficient is a statistic used to gauge the similarity of two samples.
 * Applied to strings, it can give you a value between 0 and 1 (included) which tells you how similar they are.
 * Dice coefficient is calculated by comparing the bigrams of both strings,
 * a bigram is a substring of the string of length 2.
 * read more: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
 */

// Time complexity: O(m + n), m and n being the sizes of string A and string B

// Find the bistrings of a string and return a hashmap (key => bistring, value => count)
function mapBigrams(string) {
  const bigrams = new Map();
  
  for (let i = 0; i < string.length - 1; i++) {
    const bigram = string.substring(i, i + 2);
    const count = bigrams.get(bigram) || 0;
    bigrams.set(bigram, count + 1);
  }
  
  return bigrams;
}

// Calculate the number of common bigrams between a map of bigrams and a string
function countCommonBigrams(bigrams, string) {
  let count = 0;

  for (let i = 0; i < string.length - 1; i++) {
    const bigram = string.substring(i, i + 2);
  
    if (bigrams.has(bigram)) {
      count++;
    }
  }
  
  return count;
}

// Calculate Dice coeff of 2 strings
function diceCoefficient(stringA, stringB) {
  if (stringA === stringB) {
    return 1;
  } else if (stringA.length < 2 || stringB.length < 2) {
    return 0;
  }

  const bigramsA = mapBigrams(stringA);
  const lengthA = stringA.length - 1;
  const lengthB = stringB.length - 1;

  let dice = (2 * countCommonBigrams(bigramsA, stringB)) / (lengthA + lengthB);

  // cut 0.xxxxxx to 0.xx for simplicity
  dice = Math.floor(dice * 100) / 100;

  return dice;
}

export { diceCoefficient };

// In the refactored code, we have made the following improvements to enhance maintainability:

// 1. Added appropriate comments to describe the purpose and functionality of each function.
// 2. Used meaningful variable names for better readability.
// 3. Used consistent formatting and indentation for better code organization.
// 4. Utilized default operator in the mapBigrams function to handle undefined count.
// 5. Added curly braces for if statements for better code structure.
// 6. Followed a consistent naming convention for function and variable names.
// 7. Exported the diceCoefficient function explicitly using the export statement.

// These changes improve code readability, maintainability, and organization, making it easier to understand and maintain the codebase in the future.

