// To improve the maintainability of the codebase, consider the following refactored code:

import Graph from '../../CONSTANT/javascript_algorithms/Graph';
import QuickSort from '../../Before_Refactor/javascript-algorithms/QuickSort';
import DisjointSet from '../../CONSTANT/javascript_algorithms/DisjointSet';

/**
 * @param {Graph} graph
 * @return {Graph}
 */
export default function kruskal(graph) {
  if (graph.isDirected) {
    throw new Error('Kruskal\'s algorithm works only for undirected graphs');
  }

  const minimumSpanningTree = new Graph();
  const sortingCallbacks = {
    compareCallback: (graphEdgeA, graphEdgeB) => {
      return graphEdgeA.weight - graphEdgeB.weight;
    },
  };
  const sortedEdges = new QuickSort(sortingCallbacks).sort(graph.getAllEdges());
  
  const keyCallback = (graphVertex) => graphVertex.getKey();
  const disjointSet = new DisjointSet(keyCallback);

  graph.getAllVertices().forEach((graphVertex) => {
    disjointSet.makeSet(graphVertex);
  });

  for (let edgeIndex = 0; edgeIndex < sortedEdges.length; edgeIndex += 1) {
    const currentEdge = sortedEdges[edgeIndex];

    if (!disjointSet.inSameSet(currentEdge.startVertex, currentEdge.endVertex)) {
      disjointSet.union(currentEdge.startVertex, currentEdge.endVertex);
      minimumSpanningTree.addEdge(currentEdge);
    }
  }

  return minimumSpanningTree;
}

// Here are some of the changes made to improve maintainability:

// 1. Removed unnecessary comments: Commented code should only include relevant information. Unnecessary comments clutter the codebase and can make it harder to read and maintain.

// 2. Simplified edge comparison: Instead of using a ternary operator, a comparison function was simplified to return the difference between edge weights.

// 3. Consistent error message: The error message was made consistent and clearer to reflect the functionality of the algorithm.

// 4. Removed unnecessary type definition: The type definition of the currentEdge variable was removed as it is not necessary and makes the code harder to read.

// 5. Simplified edge weight comparison: Instead of using two conditional statements, a single condition comparing the start and end vertices' disjoint sets was used.

// 6. Used a subtraction operator instead of conditional statements for edge weight comparison.

// By making these changes, the codebase becomes more maintainable and easier to understand.

