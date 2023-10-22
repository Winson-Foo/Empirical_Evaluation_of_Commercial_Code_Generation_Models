// To improve the maintainability of this codebase, we can refactor it by separating the logic into smaller, more modular functions. Additionally, we can use more descriptive variable and function names.

// Here's the refactored code:

// ```javascript
/**
 * Calculates the shortest distances from the start vertex to all other vertices
 * in the given graph using the Bellman-Ford algorithm.
 * @param {Graph} graph - The graph.
 * @param {GraphVertex} startVertex - The start vertex.
 * @returns {{distances: object, previousVertices: object}} - The distances and previous vertices.
 */
export default function bellmanFord(graph, startVertex) {
  const distances = {};
  const previousVertices = {};

  initializeDistancesAndPreviousVertices(graph, startVertex, distances, previousVertices);

  const vertices = graph.getAllVertices();
  const iterationCount = vertices.length - 1;

  for (let iteration = 0; iteration < iterationCount; iteration += 1) {
    vertices.forEach((vertex) => {
      const edges = graph.getNeighbors(vertex);

      relaxEdges(edges, vertex, distances, previousVertices);
    });
  }

  return {
    distances,
    previousVertices,
  };
}

/**
 * Initializes the distances and previousVertices objects.
 * @param {Graph} graph - The graph.
 * @param {GraphVertex} startVertex - The start vertex.
 * @param {object} distances - The distances object.
 * @param {object} previousVertices - The previous vertices object.
 */
function initializeDistancesAndPreviousVertices(graph, startVertex, distances, previousVertices) {
  graph.getAllVertices().forEach((vertex) => {
    previousVertices[vertex.getKey()] = null;

    if (vertex === startVertex) {
      distances[vertex.getKey()] = 0;
    } else {
      distances[vertex.getKey()] = Infinity;
    }
  });
}

/**
 * Relaxes all the edges of the given vertex.
 * @param {GraphEdge[]} edges - The edges.
 * @param {GraphVertex} vertex - The vertex.
 * @param {object} distances - The distances object.
 * @param {object} previousVertices - The previous vertices object.
 */
function relaxEdges(edges, vertex, distances, previousVertices) {
  edges.forEach((edge) => {
    const distanceToVertex = distances[vertex.getKey()];
    const distanceToNeighbor = distanceToVertex + edge.weight;
    const neighbor = edge.endVertex;

    if (distanceToNeighbor < distances[neighbor.getKey()]) {
      distances[neighbor.getKey()] = distanceToNeighbor;
      previousVertices[neighbor.getKey()] = vertex;
    }
  });
}
// ```

// In the refactored code, we've extracted the logic for initializing distances and previousVertices into a separate function `initializeDistancesAndPreviousVertices`. We've also extracted the logic for relaxing edges into a separate function `relaxEdges`. These functions improve readability and make the code more modular.

// Additionally, we've used more descriptive variable and function names throughout the code to improve maintainability.

