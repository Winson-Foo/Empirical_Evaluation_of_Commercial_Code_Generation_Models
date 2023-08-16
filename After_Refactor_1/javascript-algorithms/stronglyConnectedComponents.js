// To improve the maintainability of this codebase, we can make the following refactorings:

// 1. Separate the code into smaller functions with clear responsibilities.
// 2. Use meaningful variable and function names to improve code readability.
// 3. Remove unnecessary comments and clarify the remaining comments.
// 4. Use ES6 features like destructuring assignment and arrow functions to simplify the code.
// 5. Add explicit type annotations to improve code clarity.

// Here is the refactored code:

// ```javascript
import Stack from "../../../data-structures/stack/Stack";
import depthFirstSearch from "../depth-first-search/depthFirstSearch";

/**
 * Performs a depth-first search traversal on the graph and returns a stack of vertices
 * ordered by their finish time in decreasing order.
 *
 * @param {Graph} graph
 * @return {Stack}
 */
function getVerticesByFinishTime(graph) {
  const visitedVertices = new Set();
  const verticesByFinishTime = new Stack();

  graph.getAllVertices().forEach((vertex) => {
    if (!visitedVertices.has(vertex)) {
      depthFirstSearch(graph, vertex, {
        enterVertex: ({ currentVertex }) => {
          visitedVertices.add(currentVertex);
        },
        leaveVertex: ({ currentVertex }) => {
          verticesByFinishTime.push(currentVertex);
        },
        allowTraversal: ({ nextVertex }) => {
          return !visitedVertices.has(nextVertex);
        },
      });
    }
  });

  return verticesByFinishTime;
}

/**
 * Performs a depth-first search traversal on the reversed graph starting from the given vertex
 * and returns the strongly connected component set.
 *
 * @param {Graph} reversedGraph
 * @param {GraphVertex} startVertex
 * @param {Set} visitedVertices
 * @return {Array}
 */
function dfs(reversedGraph, startVertex, visitedVertices) {
  const stronglyConnectedComponent = [];
  const stack = [startVertex];

  while (stack.length > 0) {
    const currentVertex = stack.pop();

    visitedVertices.add(currentVertex);
    stronglyConnectedComponent.push(currentVertex);

    reversedGraph.getNeighbors(currentVertex).forEach((nextVertex) => {
      if (!visitedVertices.has(nextVertex)) {
        stack.push(nextVertex);
      }
    });
  }

  return stronglyConnectedComponent;
}

/**
 * Kosaraju's algorithm to find the strongly connected components in a graph.
 *
 * @param {Graph} graph
 * @return {Array}
 */
export default function stronglyConnectedComponents(graph) {
  const verticesByFinishTime = getVerticesByFinishTime(graph);
  const reversedGraph = graph.reverse();
  const visitedVertices = new Set();
  const stronglyConnectedComponents = [];

  while (!verticesByFinishTime.isEmpty()) {
    const startVertex = verticesByFinishTime.pop();

    if (!visitedVertices.has(startVertex)) {
      const stronglyConnectedComponent = dfs(reversedGraph, startVertex, visitedVertices);
      stronglyConnectedComponents.push(stronglyConnectedComponent);
    }
  }

  return stronglyConnectedComponents;
}
// ```

// Note that some assumptions were made about the implementation of the `Graph` and `Stack` classes, so please make sure to adapt the refactored code to work with your specific classes if needed.

