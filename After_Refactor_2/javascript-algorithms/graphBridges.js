// To improve the maintainability of this codebase, we can follow these steps:

// 1. Separate the helper class `VisitMetadata` into its own file and import it when needed.
// 2. Rename the `graphBridges` function to a more descriptive name like `findBridges`.
// 3. Extract the logic inside the `dfsCallbacks` object into separate functions for better readability.
// 4. Use more descriptive variable names to improve code understanding.
// 5. Remove unnecessary comments that do not add value to the code.

// Here is the refactored code with these improvements:

// File: VisitMetadata.js

/**
 * Helper class for visited vertex metadata.
 */
export default class VisitMetadata {
  constructor({ discoveryTime, lowDiscoveryTime }) {
    this.discoveryTime = discoveryTime;
    this.lowDiscoveryTime = lowDiscoveryTime;
  }
}


// File: findBridges.js
import depthFirstSearch from '../depth-first-search/depthFirstSearch';
import VisitMetadata from './VisitMetadata';

/**
 * @param {Graph} graph
 * @return {Object}
 */
export default function findBridges(graph) {
  const visitedSet = {};
  const bridges = {};
  let discoveryTime = 0;
  const startVertex = graph.getAllVertices()[0];

  const enterVertex = ({ currentVertex }) => {
    discoveryTime += 1;
    visitedSet[currentVertex.getKey()] = new VisitMetadata({
      discoveryTime,
      lowDiscoveryTime: discoveryTime,
    });
  };

  const leaveVertex = ({ currentVertex, previousVertex }) => {
    if (previousVertex === null) {
      return;
    }

    visitedSet[currentVertex.getKey()].lowDiscoveryTime = currentVertex.getNeighbors()
      .filter((neighbor) => neighbor.getKey() !== previousVertex.getKey())
      .reduce((lowestDiscoveryTime, neighbor) => {
        const neighborLowTime = visitedSet[neighbor.getKey()].lowDiscoveryTime;
        return neighborLowTime < lowestDiscoveryTime ? neighborLowTime : lowestDiscoveryTime;
      }, visitedSet[currentVertex.getKey()].lowDiscoveryTime);

    const currentLowDiscoveryTime = visitedSet[currentVertex.getKey()].lowDiscoveryTime;
    const previousLowDiscoveryTime = visitedSet[previousVertex.getKey()].lowDiscoveryTime;
    if (currentLowDiscoveryTime < previousLowDiscoveryTime) {
      visitedSet[previousVertex.getKey()].lowDiscoveryTime = currentLowDiscoveryTime;
    }

    const parentDiscoveryTime = visitedSet[previousVertex.getKey()].discoveryTime;
    if (parentDiscoveryTime < currentLowDiscoveryTime) {
      const bridge = graph.findEdge(previousVertex, currentVertex);
      bridges[bridge.getKey()] = bridge;
    }
  };

  const allowTraversal = ({ nextVertex }) => !visitedSet[nextVertex.getKey()];

  const dfsCallbacks = {
    enterVertex,
    leaveVertex,
    allowTraversal,
  };

  depthFirstSearch(graph, startVertex, dfsCallbacks);

  return bridges;
}

