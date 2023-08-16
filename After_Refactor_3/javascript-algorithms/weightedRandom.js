// To improve the maintainability of this codebase, we can make the following changes:

// 1. Use descriptive variable names: The variable names in the code can be more descriptive to improve readability and understanding of the code. For example, `items` can be renamed to `options`, `weights` can be renamed to `probabilities`, `randomNumber` can be renamed to `randomWeight`, etc.

// 2. Implement error handling using custom error classes: Instead of throwing generic `Error` instances, we can create custom error classes for specific types of errors. This will make it easier to identify and handle different types of errors that may occur.

// 3. Extract helper functions: We can extract the logic for preparing cumulative weights and picking a random item into separate helper functions. This will make the code more modular and easier to test.

// Here's the refactored code:

// ```javascript
class ItemsWeightsSizeError extends Error {
  constructor() {
    super('Items and weights must be of the same size');
  }
}

class EmptyItemsError extends Error {
  constructor() {
    super('Items must not be empty');
  }
}

function prepareCumulativeWeights(probabilities) {
  const cumulativeWeights = [];
  for (let i = 0; i < probabilities.length; i += 1) {
    cumulativeWeights[i] = probabilities[i] + (cumulativeWeights[i - 1] || 0);
  }
  return cumulativeWeights;
}

function getRandomWeight(maxCumulativeWeight) {
  return maxCumulativeWeight * Math.random();
}

function pickRandomItem(options, cumulativeWeights, randomWeight) {
  for (let itemIndex = 0; itemIndex < options.length; itemIndex += 1) {
    if (cumulativeWeights[itemIndex] >= randomWeight) {
      return {
        item: options[itemIndex],
        index: itemIndex,
      };
    }
  }
}

export default function weightedRandom(options, probabilities) {
  if (options.length !== probabilities.length) {
    throw new ItemsWeightsSizeError();
  }

  if (!options.length) {
    throw new EmptyItemsError();
  }

  const cumulativeWeights = prepareCumulativeWeights(probabilities);
  const maxCumulativeWeight = cumulativeWeights[cumulativeWeights.length - 1];
  const randomWeight = getRandomWeight(maxCumulativeWeight);

  return pickRandomItem(options, cumulativeWeights, randomWeight);
}
// ```

// In the refactored code, the custom error classes `ItemsWeightsSizeError` and `EmptyItemsError` are used to handle specific types of errors. The logic for preparing cumulative weights, generating a random weight, and picking a random item is moved to separate helper functions for better modularity. The function names and variable names are made more descriptive to improve overall code readability and maintainability.

