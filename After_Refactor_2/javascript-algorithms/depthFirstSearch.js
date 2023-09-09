// To improve the maintainability of this codebase, we can make the following changes:

// 1. Remove unnecessary variables and simplify the code:
//    - We can remove the `initiatedCallback` variable and directly assign the `callbacks` parameter to the `callbacks` variable.
//    - Instead of using an immediately invoked function expression (IIFE) to initialize the `allowTraversalCallback`, we can define it as a separate function for better readability.

// 2. Change the type of the `previousVertex` variable:
//    - Currently, the `previousVertex` variable is declared using `const` and initialized with `null`.
//    - Since its value changes during the recursive calls, it can be declared using `let` to make it mutable.

// 3. Refactor the `depthFirstSearchRecursive` function:
//    - Instead of passing `{ currentVertex, previousVertex }` as a single object to `callbacks.enterVertex` and `callbacks.leaveVertex`, we can pass them as separate parameters for better readability and usability.

// Here is the refactored code:

// ```javascript
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
 * @param {Callbacks} [callbacks]
 * @returns {Callbacks}
 */
function initCallbacks(callbacks = {}) {
  const stubCallback = () => {};

  const allowTraversalCallback = () => {
    const seen = {};
    return ({ previousVertex, currentVertex, nextVertex }) => {
      if (!seen[nextVertex.getKey()]) {
        seen[nextVertex.getKey()] = true;
        return true;
      }
      return false;
    };
  };

  return {
    allowTraversal: callbacks.allowTraversal || allowTraversalCallback(),
    enterVertex: callbacks.enterVertex || stubCallback,
    leaveVertex: callbacks.leaveVertex || stubCallback,
  };
}

/**
 * @param {Graph} graph
 * @param {GraphVertex} currentVertex
 * @param {GraphVertex} previousVertex
 * @param {Callbacks} callbacks
 */
function depthFirstSearchRecursive(graph, currentVertex, previousVertex, callbacks) {
  callbacks.enterVertex(currentVertex, previousVertex);

  graph.getNeighbors(currentVertex).forEach((nextVertex) => {
    if (callbacks.allowTraversal(previousVertex, currentVertex, nextVertex)) {
      depthFirstSearchRecursive(graph, nextVertex, currentVertex, callbacks);
    }
  });

  callbacks.leaveVertex(currentVertex, previousVertex);
}

/**
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @param {Callbacks} [callbacks]
 */
export default function depthFirstSearch(graph, startVertex, callbacks) {
  let previousVertex = null;
  depthFirstSearchRecursive(graph, startVertex, previousVertex, initCallbacks(callbacks));
}
// ```

// These changes simplify the code, make it more readable, and improve its maintainability.

