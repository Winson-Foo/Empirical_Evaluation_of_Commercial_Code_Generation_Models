// There are several steps that can be taken to improve the maintainability of this codebase:

// 1. Improve code readability by adding comments and providing clear and consistent naming conventions for variables, functions, and parameters.
// 2. Separate the code into smaller, more manageable functions that have a single responsibility.
// 3. Use descriptive function and parameter names to make the code easier to understand.
// 4. Refactor the `initCallbacks` function to use a more functional programming style and remove unnecessary variable assignments.
// 5. Move the `allowTraversalCallback` function out of `initCallbacks` to improve code organization and readability.
// 6. Use ES6 arrow functions to make the code more concise.
// 7. Add type annotations to function parameters to improve code clarity.

// Here is the refactored code:

// ```javascript
/**
 * @typedef {Object} Callbacks
 * @property {function(vertices: Object): boolean} [allowTraversal] -
 *  Determines whether DFS should traverse from the vertex to its neighbor
 *  (along the edge). By default prohibits visiting the same vertex again.
 * @property {function(vertices: Object)} [enterVertex] - Called when DFS enters the vertex.
 * @property {function(vertices: Object)} [leaveVertex] - Called when DFS leaves the vertex.
 */

/**
 * Initializes and returns the callbacks object.
 * @param {Callbacks} [callbacks]
 * @returns {Callbacks}
 */
function initCallbacks(callbacks = {}) {
  const {
    allowTraversal = ({ nextVertex }) => {
      const seen = {};
      return !seen[nextVertex.getKey()] && (seen[nextVertex.getKey()] = true);
    },
    enterVertex = () => {},
    leaveVertex = () => {}
  } = callbacks;

  return { allowTraversal, enterVertex, leaveVertex };
}

/**
 * Traverses the graph depth-first starting from the specified vertex in recursive manner.
 * @param {Graph} graph
 * @param {GraphVertex} currentVertex
 * @param {GraphVertex} previousVertex
 * @param {Callbacks} callbacks
 */
function depthFirstSearchRecursive(graph, currentVertex, previousVertex, callbacks) {
  callbacks.enterVertex({ currentVertex, previousVertex });

  graph.getNeighbors(currentVertex).forEach((nextVertex) => {
    if (callbacks.allowTraversal({ previousVertex, currentVertex, nextVertex })) {
      depthFirstSearchRecursive(graph, nextVertex, currentVertex, callbacks);
    }
  });

  callbacks.leaveVertex({ currentVertex, previousVertex });
}

/**
 * Initiates depth-first search on a graph starting from the specified vertex.
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @param {Callbacks} [callbacks]
 */
export default function depthFirstSearch(graph, startVertex, callbacks) {
  const previousVertex = null;
  depthFirstSearchRecursive(graph, startVertex, previousVertex, initCallbacks(callbacks));
} 

