// To improve the maintainability of this codebase, we can make the following refactors:

// 1. Remove the nested callback initialization and use default parameters instead.
// 2. Add type annotations to improve code readability.
// 3. Use destructuring to improve the readability of the enterVertex and leaveVertex callbacks.

// Here is the refactored code:

// ```javascript
import Queue from '../../../data-structures/queue/Queue';
import Graph from 'path-to-graph';
import GraphVertex from 'path-to-graph-vertex';

/**
 * @typedef {Object} Callbacks
 *
 * @property {function({ previousVertex: GraphVertex, currentVertex: GraphVertex, nextVertex: GraphVertex }): boolean} [allowTraversal] -
 *   Determines whether BFS should traverse from the vertex to its neighbor
 *   (along the edge). By default prohibits visiting the same vertex again.
 *
 * @property {function({ currentVertex: GraphVertex, previousVertex: GraphVertex }): void} [enterVertex] - Called when BFS enters the vertex.
 *
 * @property {function({ currentVertex: GraphVertex, previousVertex: GraphVertex }): void} [leaveVertex] - Called when BFS leaves the vertex.
 */

/**
 * @param {Callbacks} [callbacks]
 * @returns {Callbacks}
 */
function initCallbacks(callbacks = {}) {
  const {
    allowTraversal = ({ previousVertex, currentVertex, nextVertex }) => {
      const seen = {};
      if (!seen[nextVertex.getKey()]) {
        seen[nextVertex.getKey()] = true;
        return true;
      }
      return false;
    },
    enterVertex = () => {},
    leaveVertex = () => {}
  } = callbacks;

  return {
    allowTraversal,
    enterVertex,
    leaveVertex
  };
}

/**
 * @param {Graph} graph
 * @param {GraphVertex} startVertex
 * @param {Callbacks} [originalCallbacks]
 */
export default function breadthFirstSearch(graph, startVertex, originalCallbacks) {
  const callbacks = initCallbacks(originalCallbacks);
  const vertexQueue = new Queue();

  // Do initial queue setup.
  vertexQueue.enqueue(startVertex);

  let previousVertex = null;

  // Traverse all vertices from the queue.
  while (!vertexQueue.isEmpty()) {
    const currentVertex = vertexQueue.dequeue();
    callbacks.enterVertex({ currentVertex, previousVertex });

    // Add all neighbors to the queue for future traversals.
    graph.getNeighbors(currentVertex).forEach((nextVertex) => {
      if (callbacks.allowTraversal({ previousVertex, currentVertex, nextVertex })) {
        vertexQueue.enqueue(nextVertex);
      }
    });

    callbacks.leaveVertex({ currentVertex, previousVertex });

    // Memorize current vertex before next loop.
    previousVertex = currentVertex;
  }
}
// ```

// Please note that the imports for `Graph` and `GraphVertex` are not provided in the original code, so please replace the placeholders `path-to-graph` and `path-to-graph-vertex` with the actual import paths for your project.

