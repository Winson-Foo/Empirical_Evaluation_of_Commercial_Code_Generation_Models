// To improve the maintainability of the codebase, we can make the following changes:

// 1. Move the callback functions outside of the `initCallbacks` function and define them as standalone functions. This will make the code more modular and reusable.

// 2. Use the ES6 arrow function syntax for all the functions to improve readability.

// 3. Use default parameter values for the callback functions instead of creating stub functions inside `initCallbacks`.

// 4. Use more meaningful parameter names in the `depthFirstSearchRecursive` function.

// Here is the refactored code:

// ```
/**
 * @typedef {Object} Callbacks
 *
 * @property {function(vertices: Object): boolean} [allowTraversal] -
 *  Determines whether DFS should traverse from the vertex to its neighbor
 *  (along the edge). By default prohibits visiting the same vertex again.
 *
 * @property {function(vertices: Object)} [enterVertex] - Called when DFS enters the vertex.
 *
 * @property {function(vertices: Object)} [leaveVertex] - Called when DFS leaves the vertex.
 */

/**
 * Determines whether DFS should traverse from the vertex to its neighbor.
 * By default prohibits visiting the same vertex again.
 *
 * @param {Object} vertices
 * @returns {boolean}
 */
function allowTraversal(vertices) {
  const seen = {};
  return ({ nextVertex }) => {
    if (!seen[nextVertex.getKey()]) {
      seen[nextVertex.getKey()] = true;
      return true;
    }
    return false;
  };
}

/**
 * Called when DFS enters the vertex.
 *
 * @param {Object} vertices
 */
function enterVertex(vertices) {}

/**
 * Called when DFS leaves the vertex.
 *
 * @param {Object} vertices
 */
function leaveVertex(vertices) {}

/**
 * @param {Callbacks} [callbacks]
 * @returns {Callbacks}
 */
function initCallbacks(callbacks = {}) {
  return {
    allowTraversal: callbacks.allowTraversal || allowTraversal,
    enterVertex: callbacks.enterVertex || enterVertex,
    leaveVertex: callbacks.leaveVertex || leaveVertex,
  };
}

/**
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
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @param {Callbacks} [callbacks]
 */
export default function depthFirstSearch(graph, startVertex, callbacks) {
  const previousVertex = null;
  depthFirstSearchRecursive(graph, startVertex, previousVertex, initCallbacks(callbacks));
}
// ```

// By making these changes, the codebase becomes more modular, reusable, and easier to maintain.

