// To improve the maintainability of the codebase, we can follow the following suggestions:

// 1. Separate the code into smaller functions: Splitting the code into smaller functions will make it easier to understand and modify. Each function should have a clear purpose and be responsible for a specific part of the algorithm.

// 2. Use descriptive variable and function names: Clear and descriptive names for variables and functions will make the code easier to read and understand. Avoid using abbreviations or single-letter variable names.

// 3. Add comments and documentation: Comments can help explain the purpose of specific code sections or provide additional information. Adding documentation to functions can help other developers understand how to use or modify them.

// 4. Limit line length: Limit the length of each line to improve readability. Generally, it is recommended to keep lines under 80 characters.

// Here's the refactored code:

// ```javascript
import Graph from '../../../data-structures/graph/Graph';
import QuickSort from '../../sorting/quick-sort/QuickSort';
import DisjointSet from '../../../data-structures/disjoint-set/DisjointSet';

/**
 * @param {Graph} graph
 * @return {Graph}
 */
export default function kruskal(graph) {
  if (graph.isDirected) {
    throw new Error('Kruskal\'s algorithms works only for undirected graphs');
  }

  const minimumSpanningTree = new Graph();

  const sortingCallbacks = {
    compareCallback: (graphEdgeA, graphEdgeB) => {
      if (graphEdgeA.weight === graphEdgeB.weight) {
        return 1;
      }
      return graphEdgeA.weight <= graphEdgeB.weight ? -1 : 1;
    },
  };

  const sortedEdges = new QuickSort(sortingCallbacks).sort(graph.getAllEdges());

  const keyCallback = (graphVertex) => graphVertex.getKey();
  const disjointSet = new DisjointSet(keyCallback);

  createDisjointSets(graph, disjointSet);

  addEdgesToMinimumSpanningTree(sortedEdges, disjointSet, minimumSpanningTree);

  return minimumSpanningTree;
}

function createDisjointSets(graph, disjointSet) {
  graph.getAllVertices().forEach((graphVertex) => {
    disjointSet.makeSet(graphVertex);
  });
}

function addEdgesToMinimumSpanningTree(sortedEdges, disjointSet, minimumSpanningTree) {
  for (let edgeIndex = 0; edgeIndex < sortedEdges.length; edgeIndex += 1) {
    const currentEdge = sortedEdges[edgeIndex];

    if (!disjointSet.inSameSet(currentEdge.startVertex, currentEdge.endVertex)) {
      disjointSet.union(currentEdge.startVertex, currentEdge.endVertex);
      minimumSpanningTree.addEdge(currentEdge);
    }
  }
}
// ```

// By splitting the code into smaller functions and using descriptive names, it will be easier to read and understand the algorithm. The separation of concerns will also make it easier to maintain and modify the code in the future.

