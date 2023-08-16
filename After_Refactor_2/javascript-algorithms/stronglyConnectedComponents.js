// To improve the maintainability of the codebase, you can make the following changes:

// 1. Use descriptive variable and function names: Instead of using abbreviations and acronyms, use descriptive names that clearly convey the purpose of the variable or function.

// 2. Use consistent formatting and indentation: Consistent formatting and indentation improve readability and make the code easier to understand.

// 3. Break down complex functions into smaller, more manageable functions: This improves code reusability and makes it easier to understand and maintain.

// 4. Use comments to explain the code's intent and functionality: Comments help other developers understand the code and its purpose.

// Here's the refactored code with the above improvements:

import Stack from '../../../data-structures/stack/Stack';
import depthFirstSearch from '../depth-first-search/depthFirstSearch';

/**
 * Get vertices sorted by DFS finish time.
 * @param {Graph} graph
 * @return {Stack}
 */
function getVerticesSortedByDfsFinishTime(graph) {
  const visitedVerticesSet = {};
  const verticesByDfsFinishTime = new Stack();
  const notVisitedVerticesSet = {};

  graph.getAllVertices().forEach((vertex) => {
    notVisitedVerticesSet[vertex.getKey()] = vertex;
  });

  const dfsCallbacks = {
    enterVertex: ({ currentVertex }) => {
      visitedVerticesSet[currentVertex.getKey()] = currentVertex;
      delete notVisitedVerticesSet[currentVertex.getKey()];
    },
    leaveVertex: ({ currentVertex }) => {
      verticesByDfsFinishTime.push(currentVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !visitedVerticesSet[nextVertex.getKey()];
    },
  };

  while (Object.values(notVisitedVerticesSet).length) {
    const startVertexKey = Object.keys(notVisitedVerticesSet)[0];
    const startVertex = notVisitedVerticesSet[startVertexKey];
    delete notVisitedVerticesSet[startVertexKey];

    depthFirstSearch(graph, startVertex, dfsCallbacks);
  }

  return verticesByDfsFinishTime;
}

/**
 * Get strongly connected components sets.
 * @param {Graph} graph
 * @param {Stack} verticesByFinishTime
 * @return {*[]}
 */
function getSCCSets(graph, verticesByFinishTime) {
  const stronglyConnectedComponentsSets = [];
  let stronglyConnectedComponentsSet = [];
  const visitedVerticesSet = {};

  const dfsCallbacks = {
    enterVertex: ({ currentVertex }) => {
      stronglyConnectedComponentsSet.push(currentVertex);
      visitedVerticesSet[currentVertex.getKey()] = currentVertex;
    },
    leaveVertex: ({ previousVertex }) => {
      if (previousVertex === null) {
        stronglyConnectedComponentsSets.push([...stronglyConnectedComponentsSet]);
      }
    },
    allowTraversal: ({ nextVertex }) => {
      return !visitedVerticesSet[nextVertex.getKey()];
    },
  };

  while (!verticesByFinishTime.isEmpty()) {
    const startVertex = verticesByFinishTime.pop();

    stronglyConnectedComponentsSet = [];

    if (!visitedVerticesSet[startVertex.getKey()]) {
      depthFirstSearch(graph, startVertex, dfsCallbacks);
    }
  }

  return stronglyConnectedComponentsSets;
}

/**
 * Kosaraju's algorithm.
 * @param {Graph} graph
 * @return {*[]}
 */
export default function stronglyConnectedComponents(graph) {
  const verticesByFinishTime = getVerticesSortedByDfsFinishTime(graph);

  graph.reverse();

  return getSCCSets(graph, verticesByFinishTime);
}

