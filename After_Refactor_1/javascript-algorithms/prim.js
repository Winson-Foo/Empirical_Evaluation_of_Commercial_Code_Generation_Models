// To improve the maintainability of the codebase, we can follow the SOLID principles and apply some refactoring techniques. Here is the refactored code:

// ```javascript
import Graph from '../../../data-structures/graph/Graph';
import PriorityQueue from '../../../data-structures/priority-queue/PriorityQueue';

export default function prim(graph) {
  validateInputGraph(graph);

  const minimumSpanningTree = new Graph();
  const edgesQueue = new PriorityQueue();
  const visitedVertices = new Set();

  const startVertex = graph.getAllVertices()[0];
  visitedVertices.add(startVertex);

  addEdgesToQueue(startVertex.getEdges(), edgesQueue, visitedVertices);

  while (!edgesQueue.isEmpty()) {
    const currentMinEdge = edgesQueue.poll();
    const nextMinVertex = getNextMinVertex(currentMinEdge, visitedVertices);

    if (nextMinVertex) {
      minimumSpanningTree.addEdge(currentMinEdge);
      visitedVertices.add(nextMinVertex);
      addEdgesToQueue(nextMinVertex.getEdges(), edgesQueue, visitedVertices);
    }
  }

  return minimumSpanningTree;
}

function validateInputGraph(graph) {
  if (graph.isDirected) {
    throw new Error('Prim\'s algorithms works only for undirected graphs');
  }
}

function addEdgesToQueue(edges, queue, visitedVertices) {
  edges.forEach((graphEdge) => {
    const { startVertex, endVertex, weight } = graphEdge;
    if (!visitedVertices.has(startVertex) || !visitedVertices.has(endVertex)) {
      queue.add(graphEdge, weight);
    }
  });
}

function getNextMinVertex(currentEdge, visitedVertices) {
  if (!visitedVertices.has(currentEdge.startVertex)) {
    return currentEdge.startVertex;
  } else if (!visitedVertices.has(currentEdge.endVertex)) {
    return currentEdge.endVertex;
  }
  return null;
}
// ```

// In this refactored code, the following changes were made to improve maintainability:

// 1. Validations and checks are moved to separate functions for better readability and separation of concerns.
// 2. The startVertex and visitedVertices set are now initialized using the Set data structure instead of an object, providing clearer intent and simpler code.
// 3. The addEdgesToQueue function is introduced to handle adding relevant edges to the queue. This function abstracts away the logic of checking if the vertices are already visited.
// 4. The getNextMinVertex function is introduced to determine the next minimal vertex to traverse based on the current edge and visited vertices set. This function improves code readability and eliminates duplicated logic.
// 5. Proper variable and function names are used to improve code understanding and readability.

