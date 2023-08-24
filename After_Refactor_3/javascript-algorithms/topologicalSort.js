// To improve the maintainability of this codebase, you can make the following refactors:

// 1. Import only the necessary function `depthFirstSearch` instead of the entire module.
// 2. Use more descriptive variable names to improve code readability.
// 3. Instead of manipulating the `unvisitedSet` and `visitedSet` objects directly, use a `Set` data structure for better performance and readability.
// 4. Rename the `sortedStack` variable to `sortedVerticesStack` for clarity.

// Here is the refactored code:

// ```javascript
import depthFirstSearch from '../../Before_Refactor/javascript-algorithms/depthFirstSearch';

/**
 * @param {Graph} graph
 */
export default function topologicalSort(graph) {
  // Create a set of all vertices we want to visit.
  const unvisitedSet = new Set(graph.getAllVertices());

  // Create a set for all vertices that we've already visited.
  const visitedSet = new Set();

  // Create a stack of already ordered vertices.
  const sortedVerticesStack = new Stack();

  const dfsCallbacks = {
    enterVertex: ({ currentVertex }) => {
      // Add vertex to visited set in case if all its children have been explored.
      visitedSet.add(currentVertex);

      // Remove this vertex from unvisited set.
      unvisitedSet.delete(currentVertex);
    },
    leaveVertex: ({ currentVertex }) => {
      // If the vertex has been totally explored then we may push it to the stack.
      sortedVerticesStack.push(currentVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !visitedSet.has(nextVertex);
    },
  };

  // Let's go and do DFS for all unvisited nodes.
  while (unvisitedSet.size) {
    const [currentVertex] = unvisitedSet; // Get the first vertex from the unvisited set.

    // Do DFS for current vertex.
    depthFirstSearch(graph, currentVertex, dfsCallbacks);
  }

  return sortedVerticesStack.toArray();
}
// ```

// Note: Make sure to import the `Stack` and `Graph` classes from the appropriate modules.

