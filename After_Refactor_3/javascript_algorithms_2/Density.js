// To improve the maintainability of the codebase, we can make the following refactors:

// 1. Use clear and descriptive variable names: Instead of using abbreviations like "multi", use a more descriptive name like "edgeMultiplier". This makes the code easier to understand and maintain.

// 2. Add comments to explain the purpose and logic of the code: It's beneficial to add comments that explain the purpose of the function and any complex logic. This helps future developers understand the code and makes it easier to maintain.

// 3. Use arrow function syntax: Arrow functions provide a more concise and modern syntax. This can make the code more readable and easier to maintain.

// Here's the refactored code:

// ```javascript
/**
 * Calculates the density of a network.
 * Density is a measure of how many edges exist proportional to
 * how many edges would exist in a complete network.
 *
 * @param {number} numberOfNodes - The number of nodes in the network.
 * @param {number} numberOfEdges - The number of edges in the network.
 * @param {boolean} isDirected - Specifies if the network is directed. Defaults to false.
 * @returns {number} The density of the network.
 */
const density = (numberOfNodes, numberOfEdges, isDirected = false) => {
  // Calculate the multiplier based on whether the network is directed or not
  const edgeMultiplier = isDirected ? 1 : 2;

  // Calculate the density using the given formula
  return (edgeMultiplier * numberOfEdges) / (numberOfNodes * (numberOfNodes - 1));
};

export { density };
// ```

// By implementing these refactors, the code becomes more readable, self-explanatory, and easier to maintain.

