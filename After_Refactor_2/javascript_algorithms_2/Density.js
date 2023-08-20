// To improve the maintainability of the codebase, we can make several improvements:

// 1. Add meaningful comments and function/variable names to improve code readability.
// 2. Use consistent formatting, such as indentation and spacing, to make the code more visually appealing and easier to read.
// 3. Provide default values for the function parameters.
// 4. Follow best practices for variable naming and function structure.

// Here's the refactored code with the above improvements:

// ```
/**
 * Calculates the density of a network.
 * Density is measured as the ratio of the number of edges present to the number of edges that would exist in a complete network.
 * @param {number} numberOfNodes - The total number of nodes in the network.
 * @param {number} numberOfEdges - The total number of edges in the network.
 * @param {boolean} isDirected - Flag indicating if the network is directed. Default value is false.
 * @returns {number} - The density of the network.
 */
function calculateDensity(numberOfNodes, numberOfEdges, isDirected = false) {
  const multiplier = isDirected ? 1 : 2;
  const completeEdges = numberOfNodes * (numberOfNodes - 1);
  const networkDensity = (multiplier * numberOfEdges) / completeEdges;

  return networkDensity;
}

export { calculateDensity };
// ```

// By following these improvements, the code becomes more readable, maintainable, and easier to understand for future developers working on it.

