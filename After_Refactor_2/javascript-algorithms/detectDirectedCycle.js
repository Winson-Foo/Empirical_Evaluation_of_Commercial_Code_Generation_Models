// To improve the maintainability of the codebase, we can refactor it by separating the logic into smaller functions and using more descriptive variable and function names. Here is the refactored code:

// ```javascript
import depthFirstSearch from '../depth-first-search/depthFirstSearch';

/**
 * Detect cycle in directed graph using Depth First Search.
 *
 * @param {Graph} graph
 * @returns {Object|null} - The detected cycle path or null if no cycle is found.
 */
export default function detectDirectedCycle(graph) {
  let cycle = null;

  const dfsParentMap = {};
  const whiteSet = new Set();
  const graySet = new Set();
  const blackSet = new Set();

  initializeSets();

  const callbacks = {
    enterVertex: ({ currentVertex, previousVertex }) => {
      if (graySet.has(currentVertex)) {
        detectCycle(currentVertex, previousVertex);
      } else {
        setVertexStates(currentVertex, previousVertex);
      }
    },
    leaveVertex: ({ currentVertex }) => {
      moveVertexToBlackSet(currentVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !blackSet.has(nextVertex);
    },
  };

  while (whiteSet.size) {
    const startVertex = whiteSet.values().next().value;
    depthFirstSearch(graph, startVertex, callbacks);
  }

  return cycle;

  function initializeSets() {
    graph.getAllVertices().forEach((vertex) => {
      whiteSet.add(vertex);
    });
  }

  function detectCycle(currentVertex, previousVertex) {
    cycle = {};
    let currentCycleVertex = currentVertex;
    let previousCycleVertex = previousVertex;

    while (previousCycleVertex.getKey() !== currentVertex.getKey()) {
      cycle[currentCycleVertex.getKey()] = previousCycleVertex;
      currentCycleVertex = previousCycleVertex;
      previousCycleVertex = dfsParentMap[previousCycleVertex.getKey()];
    }

    cycle[currentCycleVertex.getKey()] = previousCycleVertex;
  }

  function setVertexStates(currentVertex, previousVertex) {
    graySet.add(currentVertex);
    whiteSet.delete(currentVertex);
    dfsParentMap[currentVertex.getKey()] = previousVertex;
  }

  function moveVertexToBlackSet(currentVertex) {
    blackSet.add(currentVertex);
    graySet.delete(currentVertex);
  }
}
// ```

// By separating the logic into smaller functions and using descriptive names, the code becomes easier to understand and maintain. Additionally, using sets instead of objects for the white, gray, and black sets provides improved performance and readability.

