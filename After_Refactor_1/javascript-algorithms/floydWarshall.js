// To improve the maintainability of this codebase, we can extract some of the logic into separate functions and provide better variable names to improve readability. Here is the refactored code:

// ```javascript
/**
 * @param {Graph} graph
 * @return {{distances: number[][], nextVertices: GraphVertex[][]}}
 */
export default function floydWarshall(graph) {
  const vertices = graph.getAllVertices();
  const numVertices = vertices.length;

  const nextVertices = createMatrix(numVertices, null);
  const distances = createMatrix(numVertices, Infinity);

  initializeDistancesAndNextVertices(graph, vertices, nextVertices, distances);

  for (let middleIndex = 0; middleIndex < numVertices; middleIndex++) {
    for (let startIndex = 0; startIndex < numVertices; startIndex++) {
      for (let endIndex = 0; endIndex < numVertices; endIndex++) {
        const distViaMiddle = distances[startIndex][middleIndex] + distances[middleIndex][endIndex];

        if (distances[startIndex][endIndex] > distViaMiddle) {
          distances[startIndex][endIndex] = distViaMiddle;
          nextVertices[startIndex][endIndex] = vertices[middleIndex];
        }
      }
    }
  }

  return { distances, nextVertices };
}

function createMatrix(numVertices, defaultValue) {
  return Array(numVertices).fill(null).map(() => {
    return Array(numVertices).fill(defaultValue);
  });
}

function initializeDistancesAndNextVertices(graph, vertices, nextVertices, distances) {
  vertices.forEach((startVertex, startIndex) => {
    vertices.forEach((endVertex, endIndex) => {
      if (startVertex === endVertex) {
        distances[startIndex][endIndex] = 0;
      } else {
        const edge = graph.findEdge(startVertex, endVertex);

        if (edge) {
          distances[startIndex][endIndex] = edge.weight;
          nextVertices[startIndex][endIndex] = startVertex;
        }
      }
    });
  });
}
// ```

// By extracting the initialization logic and matrix creation into separate functions, we improve code readability. Additionally, we use more descriptive variable names and make the code more modular.

