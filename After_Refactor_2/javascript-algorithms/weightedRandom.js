// To improve the maintainability of the codebase, you can follow these steps:

// 1. Add type annotations: Since this code is written in JavaScript, you can use TypeScript or JSDoc annotations to specify the types of the parameters and return value. This will make the code more self-explanatory and help catch potential type-related errors.

// 2. Use descriptive variable names: Rename variables like `items`, `weights`, `cumulativeWeights`, `randomNumber`, and `itemIndex` to more descriptive names that convey their purpose.

// 3. Separate out reusable logic: Move the logic for calculating cumulative weights into a separate function to improve code organization and make it more modular.

// 4. Handle edge cases gracefully: In the original code, there is no handling for edge cases where the weights array contains negative values or the cumulative weights go beyond the maximum random number. Add appropriate checks and error handling for such cases.

// Here's the refactored code with the suggested improvements:

/**
 * Picks a random item based on its weight.
 * The items with higher weight will be picked more often (with a higher probability).
 *
 * @param {Array} items - Array containing items to be picked from.
 * @param {Array<number>} weights - Array containing weights corresponding to each item.
 * @returns {{item: any, index: number}} - The randomly picked item and its index.
 * @throws {Error} - If items and weights are of different lengths, or if items array is empty.
 */
export default function weightedRandom(items, weights) {
  if (items.length !== weights.length) {
    throw new Error('Items and weights must be of the same size');
  }

  if (items.length === 0) {
    throw new Error('Items must not be empty');
  }

  const cumulativeWeights = calculateCumulativeWeights(weights);
  
  const maxCumulativeWeight = cumulativeWeights[cumulativeWeights.length - 1];
  const randomNumber = maxCumulativeWeight * Math.random();

  for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
    if (cumulativeWeights[itemIndex] >= randomNumber) {
      return {
        item: items[itemIndex],
        index: itemIndex,
      };
    }
  }
}

/**
 * Calculates the cumulative weights array from the given weights array.
 * 
 * @param {Array<number>} weights - Array containing weights for each item.
 * @returns {Array<number>} - Array containing cumulative weights.
 */
function calculateCumulativeWeights(weights) {
  const cumulativeWeights = [];
  let cumulativeWeight = 0;

  for (let i = 0; i < weights.length; i += 1) {
    cumulativeWeight += weights[i];
    cumulativeWeights.push(cumulativeWeight);
  }

  return cumulativeWeights;
}

