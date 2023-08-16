// Here is the refactored codebase with improved maintainability:

// ```javascript
import GraphVertex from '../../../data-structures/graph/GraphVertex';

function isAdjacent(adjacencyMatrix, verticesIndices, vertexA, vertexB) {
  const vertexAIndex = verticesIndices[vertexA.getKey()];
  const vertexBIndex = verticesIndices[vertexB.getKey()];
  return adjacencyMatrix[vertexAIndex][vertexBIndex] !== Infinity;
}

function isDuplicate(cycle, vertex) {
  return cycle.some((v) => v.getKey() === vertex.getKey());
}

function isCycle(adjacencyMatrix, verticesIndices, cycle) {
  const startVertex = cycle[0];
  const endVertex = cycle[cycle.length - 1];
  return isAdjacent(adjacencyMatrix, verticesIndices, endVertex, startVertex);
}

function hamiltonianCycleRecursive(adjacencyMatrix, vertices, verticesIndices, cycles, cycle) {
  const currentCycle = [...cycle];

  if (vertices.length === currentCycle.length) {
    if (isCycle(adjacencyMatrix, verticesIndices, currentCycle)) {
      cycles.push(currentCycle);
    }
    return;
  }

  for (let i = 0; i < vertices.length; i += 1) {
    const vertexCandidate = vertices[i];
    if (!isDuplicate(currentCycle, vertexCandidate) &&
        isAdjacent(adjacencyMatrix, verticesIndices, currentCycle[currentCycle.length - 1], vertexCandidate))
     {
      currentCycle.push(vertexCandidate);
      hamiltonianCycleRecursive(adjacencyMatrix, vertices, verticesIndices, cycles, currentCycle);
      currentCycle.pop();
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

  hamiltonianCycleRecursive(adjacencyMatrix, vertices, verticesIndices, cycles, cycle);

  return cycles;
}
// ```

// In the refactored code, I have simplified the `isSafe` and `isCycle` functions to make them more readable and remove unnecessary checks. I have also renamed some variables to improve clarity. Additionally, I have removed unnecessary comments and reordered the function parameters in `hamiltonianCycleRecursive` for consistency.

