// To improve the maintainability of this codebase, we can refactor it by breaking it down into smaller functions, introducing more descriptive variable names, and removing unnecessary comments. Here's the refactored code:

// ```javascript
import depthFirstSearch from '../depth-first-search/depthFirstSearch';

/**
 * Detect cycle in directed graph using Depth First Search.
 *
 * @param {Graph} graph
 * @returns {object} The cycle in the graph (if present)
 */
export default function detectDirectedCycle(graph) {
  let cycle = null;
  const dfsParentMap = {};
  const whiteSet = {};
  const graySet = {};
  const blackSet = {};

  initializeSets();

  const callbacks = {
    enterVertex: ({ currentVertex, previousVertex }) => {
      if (isVertexInGraySet(currentVertex)) {
        cycle = getCyclePath(currentVertex, previousVertex);
      } else {
        addToGraySet(currentVertex);
        removeFromWhiteSet(currentVertex);
        updateDfsParentMap(currentVertex, previousVertex);
      }
    },
    leaveVertex: ({ currentVertex }) => {
      addToBlackSet(currentVertex);
      removeFromGraySet(currentVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !isVertexInBlackSet(nextVertex);
    },
  };

  exploreVertices();

  return cycle;

  function initializeSets() {
    graph.getAllVertices().forEach((vertex) => {
      whiteSet[vertex.getKey()] = vertex;
    });
  }

  function isVertexInGraySet(vertex) {
    return graySet[vertex.getKey()] !== undefined;
  }

  function getCyclePath(currentVertex, previousVertex) {
    const cyclePath = {};
    let currentCycleVertex = currentVertex;
    let previousCycleVertex = previousVertex;

    while (previousCycleVertex.getKey() !== currentVertex.getKey()) {
      cyclePath[currentCycleVertex.getKey()] = previousCycleVertex;
      currentCycleVertex = previousCycleVertex;
      previousCycleVertex = dfsParentMap[previousCycleVertex.getKey()];
    }

    cyclePath[currentCycleVertex.getKey()] = previousCycleVertex;
    return cyclePath;
  }

  function addToGraySet(vertex) {
    graySet[vertex.getKey()] = vertex;
  }

  function removeFromWhiteSet(vertex) {
    delete whiteSet[vertex.getKey()];
  }

  function updateDfsParentMap(currentVertex, previousVertex) {
    dfsParentMap[currentVertex.getKey()] = previousVertex;
  }

  function addToBlackSet(vertex) {
    blackSet[vertex.getKey()] = vertex;
  }

  function removeFromGraySet(vertex) {
    delete graySet[vertex.getKey()];
  }

  function isVertexInBlackSet(vertex) {
    return blackSet[vertex.getKey()] !== undefined;
  }

  function exploreVertices() {
    while (Object.keys(whiteSet).length) {
      const firstWhiteKey = Object.keys(whiteSet)[0];
      const startVertex = whiteSet[firstWhiteKey];
      depthFirstSearch(graph, startVertex, callbacks);
    }
  }
}
// ```

// By breaking down the code into smaller functions and using descriptive variable and function names, the refactored code becomes more readable and maintainable.

