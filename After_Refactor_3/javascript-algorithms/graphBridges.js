// To improve the maintainability of this codebase, here are some refactoring suggestions:

// 1. Separate the `dfsCallbacks` object from the main code. Move it to its own function called `createDfsCallbacks()`. This will make the code more readable and modular.

// ```javascript
import depthFirstSearch from '../../Before_Refactor/javascript-algorithms/depthFirstSearch';

function createDfsCallbacks(visitedSet, bridges) {
  return {
    enterVertex: ({ currentVertex }) => {
      // ...
    },
    leaveVertex: ({ currentVertex, previousVertex }) => {
      // ...
    },
    allowTraversal: ({ nextVertex }) => {
      // ...
    },
  };
}
// ```

// 2. Move the logic for initializing the `visitedSet`, `bridges`, and `discoveryTime` variables into a separate function called `initializeData()`. This will make it easier to understand the code by separating the different responsibilities.

// ```javascript
function initializeData(graph) {
  const visitedSet = {};
  const bridges = {};
  let discoveryTime = 0;

  // Initialize visitedSet and discoveryTime here

  return { visitedSet, bridges, discoveryTime };
}
// ```

// 3. Extract the logic for finding bridges from the `leaveVertex` callback into a separate helper function called `findBridges()`. This function should take the current and previous vertices as arguments and return the bridges found.

// ```javascript
function findBridges(currentVertex, previousVertex, visitedSet, bridges) {
  // Logic for finding bridges here

  return bridges;
}
// ```

// 4. Modify the main function `graphBridges` to call the helper functions created above.

// ```javascript
export default function graphBridges(graph) {
  const { visitedSet, bridges, discoveryTime } = initializeData(graph);
  const dfsCallbacks = createDfsCallbacks(visitedSet, bridges);

  depthFirstSearch(graph, graph.getAllVertices()[0], dfsCallbacks);

  return bridges;
}
// ```

// By refactoring the code in this way, each function has a clear responsibility and can be tested independently. It also makes the code more readable and maintainable.

