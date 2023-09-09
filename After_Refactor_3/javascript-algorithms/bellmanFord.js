// To improve the maintainability of the codebase, we can follow these refactoring steps:

// 1. Split the main function into smaller, more focused functions.
// 2. Use descriptive variable and function names to improve code readability.
// 3. Abstract out repeated logic into separate functions to reduce redundancy.
// 4. Add comments to explain complex or important parts of the code.

// Here's the refactored code:

/**
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @return {{distances, previousVertices}}
 */
export default function bellmanFord(graph, startVertex) {
  const distances = {};
  const previousVertices = {};

  initializeDistancesAndPreviousVertices(graph, startVertex, distances, previousVertices);

  for (let iteration = 0; iteration < graph.getAllVertices().length - 1; iteration += 1) {
    updateDistancesAndPreviousVertices(graph, distances, previousVertices);
  }

  return {
    distances,
    previousVertices,
  };
}

function initializeDistancesAndPreviousVertices(graph, startVertex, distances, previousVertices) {
  distances[startVertex.getKey()] = 0;

  graph.getAllVertices().forEach((vertex) => {
    previousVertices[vertex.getKey()] = null;
    if (vertex.getKey() !== startVertex.getKey()) {
      distances[vertex.getKey()] = Infinity;
    }
  });
}

function updateDistancesAndPreviousVertices(graph, distances, previousVertices) {
  Object.keys(distances).forEach((vertexKey) => {
    const vertex = graph.getVertexByKey(vertexKey);

    graph.getNeighbors(vertex).forEach((neighbor) => {
      const edge = graph.findEdge(vertex, neighbor);
      const distanceToVertex = distances[vertex.getKey()];
      const distanceToNeighbor = distanceToVertex + edge.weight;
      if (distanceToNeighbor < distances[neighbor.getKey()]) {
        distances[neighbor.getKey()] = distanceToNeighbor;
        previousVertices[neighbor.getKey()] = vertex;
      }
    });
  });
}

