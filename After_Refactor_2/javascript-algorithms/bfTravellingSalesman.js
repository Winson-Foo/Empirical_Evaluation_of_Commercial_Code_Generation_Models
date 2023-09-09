// To improve the maintainability of the codebase, we can apply several refactoring techniques. Here is the refactored code:

// ```javascript
/**
 * Get all possible paths
 * @param {GraphVertex} startVertex
 * @param {GraphVertex[][]} [paths]
 * @param {GraphVertex[]} [path]
 */
function findAllPaths(startVertex, paths = [], path = []) {
  const currentPath = [...path, startVertex];
  const visitedSet = new Set(currentPath);

  const unvisitedNeighbors = startVertex.getNeighbors().filter((neighbor) => {
    return !visitedSet.has(neighbor);
  });

  if (unvisitedNeighbors.length === 0) {
    paths.push(currentPath);
    return paths;
  }

  for (const unvisitedNeighbor of unvisitedNeighbors) {
    findAllPaths(unvisitedNeighbor, paths, currentPath);
  }

  return paths;
}

/**
 * @param {number[][]} adjacencyMatrix
 * @param {object} verticesIndices
 * @param {GraphVertex[]} cycle
 * @return {number}
 */
function getCycleWeight(adjacencyMatrix, verticesIndices, cycle) {
  let weight = 0;

  for (let i = 1; i < cycle.length; i++) {
    const fromVertex = cycle[i - 1];
    const toVertex = cycle[i];
    const fromVertexIndex = verticesIndices[fromVertex.getKey()];
    const toVertexIndex = verticesIndices[toVertex.getKey()];
    weight += adjacencyMatrix[fromVertexIndex][toVertexIndex];
  }

  return weight;
}

/**
 * BRUTE FORCE approach to solve Traveling Salesman Problem.
 *
 * @param {Graph} graph
 * @return {GraphVertex[]}
 */
export default function bfTravellingSalesman(graph) {
  const startVertex = graph.getAllVertices()[0];

  const allPossiblePaths = findAllPaths(startVertex);

  const allPossibleCycles = allPossiblePaths.filter((path) => {
    const lastVertex = path[path.length - 1];
    const lastVertexNeighbors = lastVertex.getNeighbors();

    return lastVertexNeighbors.includes(startVertex);
  });

  const adjacencyMatrix = graph.getAdjacencyMatrix();
  const verticesIndices = graph.getVerticesIndices();
  let salesmanPath = [];
  let salesmanPathWeight = null;

  for (const currentCycle of allPossibleCycles) {
    const currentCycleWeight = getCycleWeight(adjacencyMatrix, verticesIndices, currentCycle);

    if (salesmanPathWeight === null || currentCycleWeight < salesmanPathWeight) {
      salesmanPath = currentCycle;
      salesmanPathWeight = currentCycleWeight;
    }
  }

  return salesmanPath;
}
// ```

// Here are the changes made:

// 1. Used ES6 features like the spread operator and arrow functions for more concise code.
// 2. Replaced the visited set implementation with a Set data structure for better performance.
// 3. Changed the `for` loop to a `for...of` loop for better readability and to avoid manual indexing.
// 4. Used more descriptive variable and function names for better clarity.
// 5. Removed unnecessary type annotations and comments.

