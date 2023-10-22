// To improve the maintainability of this codebase, I would suggest the following changes:

// 1. Organize imports: Instead of using relative paths for importing Graph and PriorityQueue, you can use absolute paths. This makes it easier to locate and manage dependencies.

// ```javascript
// import Graph from 'data-structures/graph/Graph';
// import PriorityQueue from 'data-structures/priority-queue/PriorityQueue';
// ```

// 2. Add function and parameter descriptions: It is a good practice to provide comments or JSDoc annotations to describe functions and their parameters. This improves code readability and makes it easier for other developers to understand and maintain the code.

// ```javascript
// /**
//  * Finds the minimum spanning tree of an undirected graph using Prim's algorithm.
//  *
//  * @param {Graph} graph - The graph to find the minimum spanning tree for.
//  * @return {Graph} - The minimum spanning tree of the graph.
//  */
// export default function prim(graph) {
// ```

// 3. Use descriptive variable names: Avoid using generic variable names like `graph` and `edgesQueue`. Instead, use more descriptive names like `originalGraph` and `priorityQueue`.

// 4. Extract functions for code reusability: Splitting the code into smaller functions can improve maintainability and make the code more readable. For example, you can extract the logic for adding edges to the queue into a separate function.

// ```javascript
// function addEdgesToQueue(vertex, edgesQueue, visitedVertices) {
//   vertex.getEdges().forEach((graphEdge) => {
//     if (
//       !visitedVertices[graphEdge.startVertex.getKey()] ||
//       !visitedVertices[graphEdge.endVertex.getKey()]
//     ) {
//       edgesQueue.add(graphEdge, graphEdge.weight);
//     }
//   });
// }
// ```

// 5. Reduce nested if statements: The nested if statements inside the while loop can be reduced by using early returns. This can make the code more readable and avoid deep nesting.

// ```javascript
// while (!edgesQueue.isEmpty()) {
//   const currentMinEdge = edgesQueue.poll();
//   const nextMinVertex = getNextUnvisitedVertex(currentMinEdge);
  
//   if (!nextMinVertex) {
//     continue;
//   }
  
//   minimumSpanningTree.addEdge(currentMinEdge);
//   visitedVertices[nextMinVertex.getKey()] = nextMinVertex;
  
//   addEdgesToQueue(nextMinVertex, edgesQueue, visitedVertices);
// }

// function getNextUnvisitedVertex(graphEdge) {
//   if (!visitedVertices[graphEdge.startVertex.getKey()]) {
//     return graphEdge.startVertex;
//   }
  
//   if (!visitedVertices[graphEdge.endVertex.getKey()]) {
//     return graphEdge.endVertex;
//   }
  
//   return null;
// }
// ```

// With these changes, the refactored code will look like this:

// ```javascript
import Graph from '../../CONSTANT/javascript_algorithms/Graph';
import PriorityQueue from '../../CONSTANT/javascript_algorithms/PriorityQueue';

/**
 * Finds the minimum spanning tree of an undirected graph using Prim's algorithm.
 *
 * @param {Graph} graph - The graph to find the minimum spanning tree for.
 * @return {Graph} - The minimum spanning tree of the graph.
 */
export default function prim(graph) {
  if (graph.isDirected) {
    throw new Error('Prim\'s algorithm works only for undirected graphs');
  }
  
  const minimumSpanningTree = new Graph();
  const priorityQueue = new PriorityQueue();
  const visitedVertices = {};
  
  const startVertex = graph.getAllVertices()[0];
  visitedVertices[startVertex.getKey()] = startVertex;
  
  addEdgesToQueue(startVertex, priorityQueue, visitedVertices);
  
  while (!priorityQueue.isEmpty()) {
    const currentMinEdge = priorityQueue.poll();
    const nextMinVertex = getNextUnvisitedVertex(currentMinEdge);
    
    if (!nextMinVertex) {
      continue;
    }
    
    minimumSpanningTree.addEdge(currentMinEdge);
    visitedVertices[nextMinVertex.getKey()] = nextMinVertex;
    
    addEdgesToQueue(nextMinVertex, priorityQueue, visitedVertices);
  }
  
  return minimumSpanningTree;
}

function addEdgesToQueue(vertex, priorityQueue, visitedVertices) {
  vertex.getEdges().forEach((graphEdge) => {
    if (
      !visitedVertices[graphEdge.startVertex.getKey()] ||
      !visitedVertices[graphEdge.endVertex.getKey()]
    ) {
      priorityQueue.add(graphEdge, graphEdge.weight);
    }
  });
}

function getNextUnvisitedVertex(graphEdge) {
  if (!visitedVertices[graphEdge.startVertex.getKey()]) {
    return graphEdge.startVertex;
  }
  
  if (!visitedVertices[graphEdge.endVertex.getKey()]) {
    return graphEdge.endVertex;
  }
  
  return null;
} 

