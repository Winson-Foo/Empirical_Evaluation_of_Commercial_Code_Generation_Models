// To improve the maintainability of the codebase, you can consider the following refactoring steps:

// 1. Move the `initCallbacks` function outside of the `breadthFirstSearch` function to improve readability and separation of concerns.

// ```javascript
function initCallbacks(callbacks = {}) {
  const initiatedCallbacks = {
    allowTraversal: callbacks.allowTraversal || (() => {
      const seen = {};
      return ({ nextVertex }) => {
        if (!seen[nextVertex.getKey()]) {
          seen[nextVertex.getKey()] = true;
          return true;
        }
        return false;
      };
    })(),
    enterVertex: callbacks.enterVertex || (() => {}),
    leaveVertex: callbacks.leaveVertex || (() => {}),
  };

  return initiatedCallbacks;
}

export default function breadthFirstSearch(graph, startVertex, originalCallbacks) {
  const callbacks = initCallbacks(originalCallbacks);
  // Rest of the code...
}
// ```

// 2. Use destructuring assignment to simplify the code and improve readability.

// ```javascript
export default function breadthFirstSearch(graph, startVertex, originalCallbacks) {
  const { allowTraversal, enterVertex, leaveVertex } = initCallbacks(originalCallbacks);
  const vertexQueue = new Queue();

  // Do initial queue setup.
  vertexQueue.enqueue(startVertex);

  let previousVertex = null;

  // Traverse all vertices from the queue.
  while (!vertexQueue.isEmpty()) {
    const currentVertex = vertexQueue.dequeue();
    enterVertex({ currentVertex, previousVertex });

    // Add all neighbors to the queue for future traversals.
    graph.getNeighbors(currentVertex).forEach((nextVertex) => {
      if (allowTraversal({ previousVertex, currentVertex, nextVertex })) {
        vertexQueue.enqueue(nextVertex);
      }
    });

    leaveVertex({ currentVertex, previousVertex });

    // Memorize current vertex before next loop.
    previousVertex = currentVertex;
  }
}
// ```

// These refactoring steps can help improve the maintainability of the codebase by making it more readable and easier to understand and modify in the future.

