// // To improve the maintainability of this codebase, you can consider the following refactorings:

// // 1. Extract the initialization of matrices into separate functions. This will make the code more readable and easier to understand.
// // ```javascript
// function initializeNextVertices(vertices) {
//   return Array(vertices.length).fill(null).map(() => {
//     return Array(vertices.length).fill(null);
//   });
// }

// function initializeDistances(vertices, graph) {
//   return Array(vertices.length).fill(null).map(() => {
//     return Array(vertices.length).fill(Infinity);
//   });
// }

// const nextVertices = initializeNextVertices(vertices);
// const distances = initializeDistances(vertices, graph);
// ```

// 2. Extract the logic for setting distances and nextVertices into separate functions. This will make the code more modular and reusable.
// ```javascript
// function setDistancesAndNextVertices(startVertex, startIndex, endVertex, endIndex, edge) {
//   if (startVertex === endVertex) {
//     // Distance to the vertex itself is 0.
//     distances[startIndex][endIndex] = 0;
//   } else {
//     if (edge) {
//       // There is an edge from vertex with startIndex to vertex with endIndex.
//       // Save distance and previous vertex.
//       distances[startIndex][endIndex] = edge.weight;
//       nextVertices[startIndex][endIndex] = startVertex;
//     } else {
//       distances[startIndex][endIndex] = Infinity;
//     }
//   }
// }

// vertices.forEach((startVertex, startIndex) => {
//   vertices.forEach((endVertex, endIndex) => {
//     const edge = graph.findEdge(startVertex, endVertex);
//     setDistancesAndNextVertices(startVertex, startIndex, endVertex, endIndex, edge);
//   });
// });
// ```

// 3. Extract the inner loops into a separate function. This will make the code more modular and improve readability.
// ```javascript
// function findShortestPathViaMiddleVertex(middleVertex, middleIndex) {
//   vertices.forEach((startVertex, startIndex) => {
//     vertices.forEach((endVertex, endIndex) => {
//       const distViaMiddle = distances[startIndex][middleIndex] + distances[middleIndex][endIndex];

//       if (distances[startIndex][endIndex] > distViaMiddle) {
//         distances[startIndex][endIndex] = distViaMiddle;
//         nextVertices[startIndex][endIndex] = middleVertex;
//       }
//     });
//   });
// }

// vertices.forEach((middleVertex, middleIndex) => {
//   findShortestPathViaMiddleVertex(middleVertex, middleIndex);
// });
// ```

// With these refactorings, the improved code would look like this:

// ```javascript
export default function floydWarshall(graph) {
  const vertices = graph.getAllVertices();

  function initializeNextVertices(vertices) {
    return Array(vertices.length).fill(null).map(() => {
      return Array(vertices.length).fill(null);
    });
  }

  function initializeDistances(vertices, graph) {
    return Array(vertices.length).fill(null).map(() => {
      return Array(vertices.length).fill(Infinity);
    });
  }

  function setDistancesAndNextVertices(startVertex, startIndex, endVertex, endIndex, edge) {
    if (startVertex === endVertex) {
      distances[startIndex][endIndex] = 0;
    } else {
      if (edge) {
        distances[startIndex][endIndex] = edge.weight;
        nextVertices[startIndex][endIndex] = startVertex;
      } else {
        distances[startIndex][endIndex] = Infinity;
      }
    }
  }

  function findShortestPathViaMiddleVertex(middleVertex, middleIndex) {
    vertices.forEach((startVertex, startIndex) => {
      vertices.forEach((endVertex, endIndex) => {
        const distViaMiddle = distances[startIndex][middleIndex] + distances[middleIndex][endIndex];

        if (distances[startIndex][endIndex] > distViaMiddle) {
          distances[startIndex][endIndex] = distViaMiddle;
          nextVertices[startIndex][endIndex] = middleVertex;
        }
      });
    });
  }

  const nextVertices = initializeNextVertices(vertices);
  const distances = initializeDistances(vertices, graph);

  vertices.forEach((startVertex, startIndex) => {
    vertices.forEach((endVertex, endIndex) => {
      const edge = graph.findEdge(startVertex, endVertex);
      setDistancesAndNextVertices(startVertex, startIndex, endVertex, endIndex, edge);
    });
  });

  vertices.forEach((middleVertex, middleIndex) => {
    findShortestPathViaMiddleVertex(middleVertex, middleIndex);
  });

  return { distances, nextVertices };
} 

