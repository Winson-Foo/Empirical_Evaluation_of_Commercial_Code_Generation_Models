// To improve the maintainability of the codebase, we can start by breaking down the complex functions into smaller, more manageable functions. Here is the refactored code:

// ```javascript
function clonePath(path) {
  return [...path];
}

function addToPath(vertex, path) {
  const currentPath = clonePath(path);
  currentPath.push(vertex);
  return currentPath;
}

function getVisitedSet(path) {
  return path.reduce((accumulator, vertex) => {
    const updatedAccumulator = { ...accumulator };
    updatedAccumulator[vertex.getKey()] = vertex;
    return updatedAccumulator;
  }, {});
}

function getUnvisitedNeighbors(startVertex, visitedSet) {
  return startVertex.getNeighbors().filter((neighbor) => {
    return !visitedSet[neighbor.getKey()];
  });
}

function findAllPaths(startVertex, paths = [], path = []) {
  const currentPath = addToPath(startVertex, path);
  const visitedSet = getVisitedSet(currentPath);
  const unvisitedNeighbors = getUnvisitedNeighbors(startVertex, visitedSet);

  if (unvisitedNeighbors.length === 0) {
    paths.push(currentPath);
    return paths;
  }

  for (let neighborIndex = 0; neighborIndex < unvisitedNeighbors.length; neighborIndex += 1) {
    const currentUnvisitedNeighbor = unvisitedNeighbors[neighborIndex];
    findAllPaths(currentUnvisitedNeighbor, paths, currentPath);
  }

  return paths;
}

function getCycleWeight(adjacencyMatrix, verticesIndices, cycle) {
  let weight = 0;

  for (let cycleIndex = 1; cycleIndex < cycle.length; cycleIndex += 1) {
    const fromVertex = cycle[cycleIndex - 1];
    const toVertex = cycle[cycleIndex];
    const fromVertexIndex = verticesIndices[fromVertex.getKey()];
    const toVertexIndex = verticesIndices[toVertex.getKey()];
    weight += adjacencyMatrix[fromVertexIndex][toVertexIndex];
  }

  return weight;
}

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

  for (let cycleIndex = 0; cycleIndex < allPossibleCycles.length; cycleIndex += 1) {
    const currentCycle = allPossibleCycles[cycleIndex];
    const currentCycleWeight = getCycleWeight(adjacencyMatrix, verticesIndices, currentCycle);

    if (salesmanPathWeight === null || currentCycleWeight < salesmanPathWeight) {
      salesmanPath = currentCycle;
      salesmanPathWeight = currentCycleWeight;
    }
  }

  return salesmanPath;
}

// ```

// By breaking down the code into smaller functions, we enhance the readability and maintainability of the codebase. Each function now has a well-defined purpose, making it easier to understand and modify in the future.

