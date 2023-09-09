// To improve the maintainability of the codebase, we can make the following refinements:
// - Use more descriptive variable names to enhance readability and understanding.
// - Add comments to explain the purpose and functionality of the code.

// Here's the refactored code:

// ```javascript
/*
The density of a network is a measure of how many edges exist proportional to
how many edges would exist in a complete network (where all possible edges).
https://networkx.org/documentation/networkx-1.9/reference/generated/networkx.classes.function.density.html
*/

// Calculates the density of a network
function calculateDensity(numberOfNodes, numberOfEdges, isDirected = false) {
  // Multiplier to account for directed or undirected edges
  const edgeMultiplier = isDirected ? 1 : 2;

  // Calculate the density
  const density = (edgeMultiplier * numberOfEdges) / (numberOfNodes * (numberOfNodes - 1));

  return density;
}

export { calculateDensity };
// ```

// By following a consistent coding style, using meaningful variable names, and adding appropriate comments, the refactored code becomes more readable and easier to maintain.

