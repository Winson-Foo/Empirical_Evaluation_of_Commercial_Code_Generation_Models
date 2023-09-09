// To improve the maintainability of this codebase, we can make the following refactors:

// 1. Extract the initialization of the matrices into separate functions.
// 2. Use descriptive variable names for better readability.
// 3. Split the core algorithm logic into smaller functions with descriptive names.
// 4. Use helper functions to simplify the code and reduce repetition.

// Here's the refactored code:

// ```javascript
/**
 * @param {Graph} graph
 * @return {{distances: number[][], nextVertices: GraphVertex[][]}}
 */
export default function floydWarshall(graph) {
  const vertices = graph.getAllVertices();
  const numVertices = vertices.length;
  const distances = initializeDistances(vertices);
  const nextVertices = initializeNextVertices(vertices);

  calculateShortestPaths(vertices, distances, nextVertices, graph);

  return { distances, nextVertices };
}

function initializeDistances(vertices) {
  const distances = Array(vertices.length).fill(null).map(() => Array(vertices.length).fill(Infinity));
  
  vertices.forEach((startVertex, startIndex) => {
    distances[startIndex][startIndex] = 0;
    const edges = graph.getEdges(startVertex);

    edges.forEach(edge => {
      const endIndex = vertices.indexOf(edge.endVertex);
      distances[startIndex][endIndex] = edge.weight;
    });
  });

  return distances;
}

function initializeNextVertices(vertices) {
  const nextVertices = Array(vertices.length).fill(null).map(() => Array(vertices.length).fill(null));
  
  vertices.forEach((startVertex, startIndex) => {
    const edges = graph.getEdges(startVertex);

    edges.forEach(edge => {
      const endIndex = vertices.indexOf(edge.endVertex);
      nextVertices[startIndex][endIndex] = startVertex;
    });
  });

  return nextVertices;
}

function calculateShortestPaths(vertices, distances, nextVertices, graph) {
  vertices.forEach((middleVertex, middleIndex) => {
    vertices.forEach((startVertex, startIndex) => {
      vertices.forEach((endVertex, endIndex) => {
        const distViaMiddle = distances[startIndex][middleIndex] + distances[middleIndex][endIndex];

        if (distances[startIndex][endIndex] > distViaMiddle) {
          distances[startIndex][endIndex] = distViaMiddle;
          nextVertices[startIndex][endIndex] = middleVertex;
        }
      });
    });
  });
}
// ```

// By extracting the initialization of the matrices into separate functions and splitting the core algorithm logic into smaller functions, the code becomes more maintainable and easier to understand. Additionally, using descriptive variable names and helper functions helps improve the readability of the code.

