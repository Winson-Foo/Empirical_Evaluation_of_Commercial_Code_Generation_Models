// To improve the maintainability of the codebase, you can consider the following refactored code:

// ```javascript
/**
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @return {{distances, previousVertices}}
 */
export default function bellmanFord(graph, startVertex) {
  const distances = {};
  const previousVertices = {};

  initDistances(graph, startVertex, distances, previousVertices);
  iterateVertices(graph, distances, previousVertices);

  return {
    distances,
    previousVertices,
  };
}

function initDistances(graph, startVertex, distances, previousVertices) {
  graph.getAllVertices().forEach((vertex) => {
    distances[vertex.getKey()] = Infinity;
    previousVertices[vertex.getKey()] = null;
  });

  distances[startVertex.getKey()] = 0;
}

function iterateVertices(graph, distances, previousVertices) {
  const vertices = graph.getAllVertices();
  const vertexCount = vertices.length;

  for (let iteration = 0; iteration < vertexCount - 1; iteration += 1) {
    vertices.forEach((vertex) => {
      const neighbors = graph.getNeighbors(vertex);

      neighbors.forEach((neighbor) => {
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
}
// ```

// In the refactored code:
// 1. The logic for initializing distances and previousVertices has been moved to a separate function `initDistances()`.
// 2. The logic for iterating over the vertices has been moved to a separate function `iterateVertices()`.
// 3. The code is split into smaller, more focused functions, improving readability and maintainability.
// 4. Variable names have been updated to be more descriptive.
// 5. An intermediate variable `vertexCount` is used to avoid repeated calls to `graph.getAllVertices().length`.
// 6. Overall, the code is more modular and easier to understand and maintain.

