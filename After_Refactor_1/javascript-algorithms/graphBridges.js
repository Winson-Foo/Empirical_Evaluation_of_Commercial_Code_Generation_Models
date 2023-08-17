// To improve the maintainability of this codebase, here are some suggested refactors:

// 1. Separate the `VisitMetadata` class into its own file, `VisitMetadata.js`:
// ```javascript
// // VisitMetadata.js
// export default class VisitMetadata {
//   constructor({ discoveryTime, lowDiscoveryTime }) {
//     this.discoveryTime = discoveryTime;
//     this.lowDiscoveryTime = lowDiscoveryTime;
//   }
// }
// ```

// 2. Use named imports instead of default import for `depthFirstSearch`:
// ```javascript
// import { depthFirstSearch } from '../depth-first-search/depthFirstSearch';
// ```

// 3. Use arrow function syntax for the callback functions in `dfsCallbacks`:
// ```javascript
// const dfsCallbacks = {
//   enterVertex: ({ currentVertex }) => {
//     // ...
//   },
//   leaveVertex: ({ currentVertex, previousVertex }) => {
//     // ...
//   },
//   allowTraversal: ({ nextVertex }) => {
//     // ...
//   },
// };
// ```

// 4. Extract some logic into separate functions to improve readability:
// ```javascript
// function updateLowDiscoveryTime(currentVertex, previousVertex) {
//   visitedSet[currentVertex.getKey()].lowDiscoveryTime = currentVertex.getNeighbors()
//     .filter((earlyNeighbor) => earlyNeighbor.getKey() !== previousVertex.getKey())
//     .reduce((lowestDiscoveryTime, neighbor) => {
//       const neighborLowTime = visitedSet[neighbor.getKey()].lowDiscoveryTime;
//       return Math.min(neighborLowTime, lowestDiscoveryTime);
//     }, visitedSet[currentVertex.getKey()].lowDiscoveryTime);
// }

// function updatePreviousVertexLowTime(currentVertex, previousVertex) {
//   const currentLowDiscoveryTime = visitedSet[currentVertex.getKey()].lowDiscoveryTime;
//   const previousLowDiscoveryTime = visitedSet[previousVertex.getKey()].lowDiscoveryTime;
//   if (currentLowDiscoveryTime < previousLowDiscoveryTime) {
//     visitedSet[previousVertex.getKey()].lowDiscoveryTime = currentLowDiscoveryTime;
//   }
// }

// function checkArticulationPoint(currentVertex, previousVertex) {
//   const parentDiscoveryTime = visitedSet[previousVertex.getKey()].discoveryTime;
//   const currentLowDiscoveryTime = visitedSet[currentVertex.getKey()].lowDiscoveryTime;
//   if (parentDiscoveryTime < currentLowDiscoveryTime) {
//     const bridge = graph.findEdge(previousVertex, currentVertex);
//     bridges[bridge.getKey()] = bridge;
//   }
// }

// const dfsCallbacks = {
//   enterVertex: ({ currentVertex }) => {
//     discoveryTime += 1;
//     visitedSet[currentVertex.getKey()] = new VisitMetadata({
//       discoveryTime,
//       lowDiscoveryTime: discoveryTime,
//     });
//   },
//   leaveVertex: ({ currentVertex, previousVertex }) => {
//     if (previousVertex === null) {
//       return;
//     }
//     updateLowDiscoveryTime(currentVertex, previousVertex);
//     updatePreviousVertexLowTime(currentVertex, previousVertex);
//     checkArticulationPoint(currentVertex, previousVertex);
//   },
//   allowTraversal: ({ nextVertex }) => {
//     return !visitedSet[nextVertex.getKey()];
//   },
// };
// ```

// The refactored code is as follows:
// ```javascript
import { depthFirstSearch } from '../depth-first-search/depthFirstSearch';
import VisitMetadata from './VisitMetadata';

function updateLowDiscoveryTime(currentVertex, previousVertex) {
  visitedSet[currentVertex.getKey()].lowDiscoveryTime = currentVertex.getNeighbors()
    .filter((earlyNeighbor) => earlyNeighbor.getKey() !== previousVertex.getKey())
    .reduce((lowestDiscoveryTime, neighbor) => {
      const neighborLowTime = visitedSet[neighbor.getKey()].lowDiscoveryTime;
      return Math.min(neighborLowTime, lowestDiscoveryTime);
    }, visitedSet[currentVertex.getKey()].lowDiscoveryTime);
}

function updatePreviousVertexLowTime(currentVertex, previousVertex) {
  const currentLowDiscoveryTime = visitedSet[currentVertex.getKey()].lowDiscoveryTime;
  const previousLowDiscoveryTime = visitedSet[previousVertex.getKey()].lowDiscoveryTime;
  if (currentLowDiscoveryTime < previousLowDiscoveryTime) {
    visitedSet[previousVertex.getKey()].lowDiscoveryTime = currentLowDiscoveryTime;
  }
}

function checkArticulationPoint(currentVertex, previousVertex) {
  const parentDiscoveryTime = visitedSet[previousVertex.getKey()].discoveryTime;
  const currentLowDiscoveryTime = visitedSet[currentVertex.getKey()].lowDiscoveryTime;
  if (parentDiscoveryTime < currentLowDiscoveryTime) {
    const bridge = graph.findEdge(previousVertex, currentVertex);
    bridges[bridge.getKey()] = bridge;
  }
}

export default function graphBridges(graph) {
  const visitedSet = {};
  const bridges = {};
  let discoveryTime = 0;

  const startVertex = graph.getAllVertices()[0];

  const dfsCallbacks = {
    enterVertex: ({ currentVertex }) => {
      discoveryTime += 1;
      visitedSet[currentVertex.getKey()] = new VisitMetadata({
        discoveryTime,
        lowDiscoveryTime: discoveryTime,
      });
    },
    leaveVertex: ({ currentVertex, previousVertex }) => {
      if (previousVertex === null) {
        return;
      }
      updateLowDiscoveryTime(currentVertex, previousVertex);
      updatePreviousVertexLowTime(currentVertex, previousVertex);
      checkArticulationPoint(currentVertex, previousVertex);
    },
    allowTraversal: ({ nextVertex }) => {
      return !visitedSet[nextVertex.getKey()];
    },
  };

  depthFirstSearch(graph, startVertex, dfsCallbacks);

  return bridges;
}

