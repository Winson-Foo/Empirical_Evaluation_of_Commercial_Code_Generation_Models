// // To improve the maintainability of this codebase, here are some suggestions and the refactored code:

// // 1. Divide the code into smaller, more manageable functions: Split the code into smaller functions, each responsible for a specific task. This will make the code easier to understand and maintain.

// // Refactored code:
// // ```javascript
// function getAllPossiblePaths(startVertex) {
//   return findAllPaths(startVertex);
// }

// function filterCycles(startVertex, allPossiblePaths) {
//   return allPossiblePaths.filter((path) => {
//     const lastVertex = path[path.length - 1];
//     const lastVertexNeighbors = lastVertex.getNeighbors();

//     return lastVertexNeighbors.includes(startVertex);
//   });
// }

// function findOptimalCycle(adjacencyMatrix, verticesIndices, allPossibleCycles) {
//   let salesmanPath = [];
//   let salesmanPathWeight = null;
//   for (let cycleIndex = 0; cycleIndex < allPossibleCycles.length; cycleIndex += 1) {
//     const currentCycle = allPossibleCycles[cycleIndex];
//     const currentCycleWeight = getCycleWeight(adjacencyMatrix, verticesIndices, currentCycle);

//     if (salesmanPathWeight === null || currentCycleWeight < salesmanPathWeight) {
//       salesmanPath = currentCycle;
//       salesmanPathWeight = currentCycleWeight;
//     }
//   }

//   return salesmanPath;
// }

// export default function bfTravellingSalesman(graph) {
//   const startVertex = graph.getAllVertices()[0];
//   const allPossiblePaths = getAllPossiblePaths(startVertex);
//   const allPossibleCycles = filterCycles(startVertex, allPossiblePaths);
//   const adjacencyMatrix = graph.getAdjacencyMatrix();
//   const verticesIndices = graph.getVerticesIndices();
  
//   return findOptimalCycle(adjacencyMatrix, verticesIndices, allPossibleCycles);
// }
// ```

// 2. Use meaningful variable and function names: Rename variables and functions to more descriptive names that accurately reflect their purpose. This will make the code easier to understand and maintain.

// Refactored code:
// ```javascript
function getAllPossiblePaths(startVertex) {
  return findAllPaths(startVertex);
}

function filterCycles(startVertex, allPossiblePaths) {
  return allPossiblePaths.filter((path) => {
    const lastVertex = path[path.length - 1];
    const lastVertexNeighbors = lastVertex.getNeighbors();

    return lastVertexNeighbors.includes(startVertex);
  });
}

function findOptimalCycle(adjacencyMatrix, verticesIndices, allPossibleCycles) {
  let optimalCycle = [];
  let optimalCycleWeight = null;
  for (let cycleIndex = 0; cycleIndex < allPossibleCycles.length; cycleIndex += 1) {
    const currentCycle = allPossibleCycles[cycleIndex];
    const currentCycleWeight = getCycleWeight(adjacencyMatrix, verticesIndices, currentCycle);

    if (optimalCycleWeight === null || currentCycleWeight < optimalCycleWeight) {
      optimalCycle = currentCycle;
      optimalCycleWeight = currentCycleWeight;
    }
  }

  return optimalCycle;
}

export default function bfTravellingSalesman(graph) {
  const startVertex = graph.getAllVertices()[0];
  const allPossiblePaths = getAllPossiblePaths(startVertex);
  const allPossibleCycles = filterCycles(startVertex, allPossiblePaths);
  const adjacencyMatrix = graph.getAdjacencyMatrix();
  const verticesIndices = graph.getVerticesIndices();
  
  return findOptimalCycle(adjacencyMatrix, verticesIndices, allPossibleCycles);
}
// ```

// These suggestions and refactored code should make the codebase more maintainable by improving readability and reducing complexity.

