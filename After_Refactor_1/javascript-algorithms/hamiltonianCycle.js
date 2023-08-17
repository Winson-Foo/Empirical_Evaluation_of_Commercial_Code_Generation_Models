// To improve the maintainability of this codebase, we can make the following changes:

// 1. Avoid using hardcoded values and magic numbers: Instead of using `Infinity` as a special value in the adjacency matrix, we can define a constant variable to represent an infinite value. This makes the code more readable and easier to understand.

// ```javascript
// const INFINITY = Infinity;

// ...

// if (adjacencyMatrix[endVertexAdjacencyIndex][candidateVertexAdjacencyIndex] === INFINITY) {
//   return false;
// }

// ...

// return adjacencyMatrix[endVertexAdjacencyIndex][startVertexAdjacencyIndex] !== INFINITY;
// ```

// 2. Extract reusable functions: We can extract some of the functionalities into separate functions to improve code readability and reusability. For example, we can extract the logic of checking for duplicate vertices in a cycle into a separate function.

// ```javascript
// function hasDuplicateVertex(cycle, vertex) {
//   return cycle.some((v) => v.getKey() === vertex.getKey());
// }

// ...

// // Check if vertexCandidate is being added to the path for the first time.
// const candidateDuplicate = hasDuplicateVertex(cycle, vertexCandidate);

// return !candidateDuplicate;
// ```

// 3. Use clear and descriptive variable names: Instead of using variable names like `currentCycle` and `vertexCandidate`, we can use more descriptive names that clearly convey their purpose and meaning, such as `path` and `currentVertex`. This makes the code easier to understand and maintain.

// ```javascript
// ...

// // Clone cycle in order to prevent it from modification by other DFS branches.
// const path = [...cycle].map((vertex) => new GraphVertex(vertex.value));

// ...

// for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex += 1) {
//   // Get current vertex that we will try to put into next path step and see if it fits.
//   const currentVertex = vertices[vertexIndex];

//   ...

//   // Add current vertex to path.
//   path.push(currentVertex);

//   ...

//   // Remove current vertex from path in order to try another one.
//   path.pop();
// }

// ...

// ```

// Here is the refactored code:

// ```javascript
import GraphVertex from '../../../data-structures/graph/GraphVertex';

const INFINITY = Infinity;

function isSafe(adjacencyMatrix, verticesIndices, cycle, vertexCandidate) {
  const endVertex = cycle[cycle.length - 1];

  const candidateVertexAdjacencyIndex = verticesIndices[vertexCandidate.getKey()];
  const endVertexAdjacencyIndex = verticesIndices[endVertex.getKey()];

  if (adjacencyMatrix[endVertexAdjacencyIndex][candidateVertexAdjacencyIndex] === INFINITY) {
    return false;
  }

  const candidateDuplicate = cycle.some((vertex) => vertex.getKey() === vertexCandidate.getKey());

  return !candidateDuplicate;
}

function hasDuplicateVertex(cycle, vertex) {
  return cycle.some((v) => v.getKey() === vertex.getKey());
}

function isCycle(adjacencyMatrix, verticesIndices, cycle) {
  const startVertex = cycle[0];
  const endVertex = cycle[cycle.length - 1];

  const startVertexAdjacencyIndex = verticesIndices[startVertex.getKey()];
  const endVertexAdjacencyIndex = verticesIndices[endVertex.getKey()];

  return adjacencyMatrix[endVertexAdjacencyIndex][startVertexAdjacencyIndex] !== INFINITY;
}

function hamiltonianCycleRecursive({
  adjacencyMatrix,
  vertices,
  verticesIndices,
  cycles,
  cycle,
}) {
  const path = [...cycle].map((vertex) => new GraphVertex(vertex.value));

  if (vertices.length === path.length) {
    if (isCycle(adjacencyMatrix, verticesIndices, path)) {
      cycles.push(path);
    }
    return;
  }

  for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex += 1) {
    const currentVertex = vertices[vertexIndex];

    if (isSafe(adjacencyMatrix, verticesIndices, path, currentVertex)) {
      path.push(currentVertex);

      hamiltonianCycleRecursive({
        adjacencyMatrix,
        vertices,
        verticesIndices,
        cycles,
        cycle: path,
      });

      path.pop();
    }
  }
}

export default function hamiltonianCycle(graph) {
  const verticesIndices = graph.getVerticesIndices();
  const adjacencyMatrix = graph.getAdjacencyMatrix();
  const vertices = graph.getAllVertices();

  const startVertex = vertices[0];

  const cycles = [];

  const cycle = [startVertex];

  hamiltonianCycleRecursive({
    adjacencyMatrix,
    vertices,
    verticesIndices,
    cycles,
    cycle,
  });

  return cycles;
}

