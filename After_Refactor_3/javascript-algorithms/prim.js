// To improve the maintainability of this codebase, we can do the following steps:

// 1. Create a separate class or module for the `prim` function, instead of using a default export. This will make it easier to import and use the function in other parts of the codebase.

// 2. Add comments and documentation to explain the purpose of each section of code. This will make it easier for other developers to understand and maintain the code in the future.

// 3. Create helper functions or methods for repetitive tasks. For example, we can create a `getUnvisitedVertex` function to find the next unvisited minimal vertex to traverse. This will make the code more readable and reduce code duplication.

// 4. Use more descriptive variable names to improve code readability. For example, instead of `graph`, we can use `originalGraph` to make it clear that it is the original graph passed as an argument.

// Here is the refactored code with the suggested improvements:

// ```javascript
import Graph from '../../../data-structures/graph/Graph';
import PriorityQueue from '../../../data-structures/priority-queue/PriorityQueue';

export class PrimAlgorithm {
  /**
   * @param {Graph} originalGraph
   * @return {Graph}
   */
  static getMinimumSpanningTree(originalGraph) {
    if (originalGraph.isDirected) {
      throw new Error('Prim\'s algorithm works only for undirected graphs');
    }

    const minimumSpanningTree = new Graph();
    const edgesQueue = new PriorityQueue();
    const visitedVertices = {};

    const startVertex = originalGraph.getAllVertices()[0];
    visitedVertices[startVertex.getKey()] = startVertex;

    startVertex.getEdges().forEach((graphEdge) => {
      edgesQueue.add(graphEdge, graphEdge.weight);
    });

    while (!edgesQueue.isEmpty()) {
      const currentMinEdge = edgesQueue.poll();
      const nextMinVertex = PrimAlgorithm.getUnvisitedVertex(currentMinEdge, visitedVertices);

      if (nextMinVertex) {
        minimumSpanningTree.addEdge(currentMinEdge);
        visitedVertices[nextMinVertex.getKey()] = nextMinVertex;

        nextMinVertex.getEdges().forEach((graphEdge) => {
          if (
            !visitedVertices[graphEdge.startVertex.getKey()]
            || !visitedVertices[graphEdge.endVertex.getKey()]
          ) {
            edgesQueue.add(graphEdge, graphEdge.weight);
          }
        });
      }
    }

    return minimumSpanningTree;
  }

  /**
   * @param {GraphEdge} graphEdge
   * @param {Object} visitedVertices
   * @return {GraphVertex}
   */
  static getUnvisitedVertex(graphEdge, visitedVertices) {
    if (!visitedVertices[graphEdge.startVertex.getKey()]) {
      return graphEdge.startVertex;
    } else if (!visitedVertices[graphEdge.endVertex.getKey()]) {
      return graphEdge.endVertex;
    }

    return null;
  }
}
// ```

// Now you can import the `PrimAlgorithm` class and use the `getMinimumSpanningTree` method to get the minimum spanning tree of a given graph:

// ```javascript
import { PrimAlgorithm } from './PrimAlgorithm';
import Graph from '../../../data-structures/graph/Graph';

const graph = new Graph();
// Add vertices and edges to the graph

const minimumSpanningTree = PrimAlgorithm.getMinimumSpanningTree(graph);
// Use the minimum spanning tree
// ```

// Note: The refactored code is assuming that the `Graph` and `PriorityQueue` classes are correctly implemented and imported from the respective locations. The refactored code only focuses on improving the maintainability of the existing codebase.

