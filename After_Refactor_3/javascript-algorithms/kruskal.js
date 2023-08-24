// To improve the maintainability of the codebase, you can consider the following steps:

// 1. Separate the code into smaller functions: Break down the code into smaller functions with specific responsibilities. This will make the code easier to read, understand, and maintain. For example, you can create separate functions for sorting the edges, creating disjoint sets, and adding edges to the minimum spanning tree.

// 2. Use meaningful variable and function names: Choose descriptive names for variables and functions that accurately reflect their purpose. This will make the code self-explanatory and easier to understand.

// 3. Remove magic numbers and strings: Instead of hardcoding values like weights and error messages, define them as constants or variables. This will make the code more flexible and maintainable, as changes can be made in a centralized location.

// 4. Add comments and documentation: Include comments and documentation to provide explanations for the code logic, algorithm steps, and input/output requirements. This will aid in understanding the code and troubleshooting any issues.

// Here is the refactored code with the above improvements:

import Graph from '../../CONSTANT/javascript-algorithms/Graph';
import QuickSort from '../../Before_Refactor/javascript-algorithms/QuickSort';
import DisjointSet from '../../CONSTANT/javascript-algorithms/DisjointSet';

/**
 * Calculates the minimum spanning tree of an undirected graph using Kruskal's algorithm.
 * @param {Graph} graph - The input graph.
 * @returns {Graph} - The minimum spanning tree of the input graph.
 */
export default function kruskal(graph) {
  // Throw an error if the graph is directed since the algorithm works only for undirected graphs.
  if (graph.isDirected) {
    throw new Error('Kruskal\'s algorithms works only for undirected graphs');
  }

  // Initialize a new graph that will contain the minimum spanning tree of the original graph.
  const minimumSpanningTree = new Graph();

  // Sort all graph edges in increasing order.
  const sortedEdges = sortGraphEdgesByWeight(graph.getAllEdges());

  // Create disjoint sets for all graph vertices.
  const disjointSet = createDisjointSets(graph);

  // Go through all edges starting from the minimum one and try to add them to the minimum spanning tree.
  for (const edge of sortedEdges) {
    if (!disjointSet.isSameSet(edge.startVertex, edge.endVertex)) {
      disjointSet.union(edge.startVertex, edge.endVertex);
      minimumSpanningTree.addEdge(edge);
    }
  }

  return minimumSpanningTree;
}

function sortGraphEdgesByWeight(edges) {
  const sortingCallbacks = {
    compareCallback: (edgeA, edgeB) => edgeA.weight - edgeB.weight,
  };
  return new QuickSort(sortingCallbacks).sort(edges);
}

function createDisjointSets(graph) {
  const keyCallback = (vertex) => vertex.getKey();
  const disjointSet = new DisjointSet(keyCallback);

  for (const vertex of graph.getAllVertices()) {
    disjointSet.makeSet(vertex);
  }

  return disjointSet;
}

// Example usage:
const graph = new Graph();
// Add vertices and edges to the graph...

const minimumSpanningTree = kruskal(graph);
console.log(minimumSpanningTree.getAllEdges());

