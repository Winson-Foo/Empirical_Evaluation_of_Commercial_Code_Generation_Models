// To improve the maintainability of the codebase, we can make the following changes:

// 1. Move the logic for creating the unvisitedSet and visitedSet to a separate function, which will make it easier to understand and maintain the code.

// 2. Rename the dfsCallbacks object to a more descriptive name, such as "dfsCallbacksHandlers".

// 3. Instead of using Object.keys(unvisitedSet)[0] to get the first vertex in the unvisitedSet, we can use the Array.prototype.shift() method, which will remove and return the first element from an array.

// 4. Add comments to explain the purpose of each section of code.

// Here is the refactored code:

import Stack from '../../CONSTANT/javascript-algorithms/Stack';
import depthFirstSearch from '../../Before_Refactor/javascript-algorithms/depthFirstSearch';

/**
 * @param {Graph} graph
 */
export default function topologicalSort(graph) {
  const unvisitedSet = createUnvisitedSet(graph);
  const visitedSet = {};
  const sortedStack = new Stack();

  const dfsCallbacksHandlers = {
    enterVertex: ({ currentVertex }) => {
      visitedSet[currentVertex.getKey()] = currentVertex;
      delete unvisitedSet[currentVertex.getKey()];
    },
    leaveVertex: ({ currentVertex }) => {
      sortedStack.push(currentVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !visitedSet[nextVertex.getKey()];
    },
  };

  while (Object.keys(unvisitedSet).length) {
    const currentVertexKey = Object.keys(unvisitedSet).shift();
    const currentVertex = unvisitedSet[currentVertexKey];
    depthFirstSearch(graph, currentVertex, dfsCallbacksHandlers);
  }

  return sortedStack.toArray();
}

function createUnvisitedSet(graph) {
  const unvisitedSet = {};
  graph.getAllVertices().forEach((vertex) => {
    unvisitedSet[vertex.getKey()] = vertex;
  });
  return unvisitedSet;
}

