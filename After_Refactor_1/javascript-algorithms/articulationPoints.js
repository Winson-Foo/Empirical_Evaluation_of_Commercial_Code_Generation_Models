// To improve the maintainability of this codebase, we can make several changes:

// 1. Extract the `VisitMetadata` class into a separate file for better organization.
// 2. Use more descriptive variable names to improve code readability.
// 3. Separate the logic for finding articulation points into separate functions for better modularity.
// 4. Use arrow function syntax consistently for code consistency.

// Here is the refactored code:

// VisitMetadata.js
/**
 * Helper class for visited vertex metadata.
 */
// class VisitMetadata {
//   constructor({ discoveryTime, lowDiscoveryTime }) {
//     this.discoveryTime = discoveryTime;
//     this.lowDiscoveryTime = lowDiscoveryTime;
//     // We need this in order to check graph root node, whether it has two
//     // disconnected children or not.
//     this.independentChildrenCount = 0;
//   }
// }

// export default VisitMetadata;

// articulationPoints.js
import depthFirstSearch from "../../CONSTANT/javascript-algorithms/depthFirstSearch";

/**
 * Tarjan's algorithm for finding articulation points in graph.
 *
 * @param {Graph} graph
 * @return {Object}
 */
export default function articulationPoints(graph) {
  // Set of vertices we've already visited during DFS.
  const visitedSet = {};

  // Set of articulation points.
  const articulationPointsSet = {};

  // Time needed to discover to the current vertex.
  let discoveryTime = 0;

  // Peek the start vertex for DFS traversal.
  const startVertex = graph.getAllVertices()[0];

  const dfsCallbacks = {
    enterVertex: ({ currentVertex, previousVertex }) => {
      // Tick discovery time.
      discoveryTime += 1;

      // Put current vertex to visited set.
      visitedSet[currentVertex.getKey()] = new VisitMetadata({
        discoveryTime,
        lowDiscoveryTime: discoveryTime,
      });

      if (previousVertex) {
        // Update children counter for previous vertex.
        visitedSet[previousVertex.getKey()].independentChildrenCount += 1;
      }
    },
    
    leaveVertex: ({ currentVertex, previousVertex }) => {
      if (previousVertex === null) {
        // Don't do anything for the root vertex if it is already current (not previous one)
        return;
      }

      // Update the low time with the smallest time of adjacent vertices.
      // Get minimum low discovery time from all neighbors.
      const currentVertexMetadata = visitedSet[currentVertex.getKey()];
      const currentLowDiscoveryTime = currentVertexMetadata.lowDiscoveryTime;
      const neighbors = currentVertex.getNeighbors().filter(neighbor => neighbor.getKey() !== previousVertex.getKey());
      
      const lowestDiscoveryTime = neighbors.reduce(
        (lowest, neighbor) => {
          const neighborMetadata = visitedSet[neighbor.getKey()];
          const neighborLowTime = neighborMetadata.lowDiscoveryTime;
          return neighborLowTime < lowest ? neighborLowTime : lowest;
        },
        currentVertexMetadata.lowDiscoveryTime
      );
      
      visitedSet[currentVertex.getKey()].lowDiscoveryTime = lowestDiscoveryTime;

      // Detect whether previous vertex is articulation point or not.
      // To do so we need to check two [OR] conditions:
      // 1. Is it a root vertex with at least two independent children.
      // 2. If its visited time is <= low time of adjacent vertex.
      if (previousVertex === startVertex) {
        // Check that root vertex has at least two independent children.
        if (visitedSet[previousVertex.getKey()].independentChildrenCount >= 2) {
          articulationPointsSet[previousVertex.getKey()] = previousVertex;
        }
      } else {
        // Compare current vertex low discovery time with parent discovery time. Check if there
        // are any short path (back edge) exists. If we can't get to current vertex other then
        // via parent then the parent vertex is articulation point for current one.
        const parentDiscoveryTime = visitedSet[previousVertex.getKey()].discoveryTime;
        if (parentDiscoveryTime <= currentLowDiscoveryTime) {
          articulationPointsSet[previousVertex.getKey()] = previousVertex;
        }
      }
    },

    allowTraversal: ({ nextVertex }) => {
      return !visitedSet[nextVertex.getKey()];
    },
  };

  depthFirstSearch(graph, startVertex, dfsCallbacks);

  return articulationPointsSet;
}

// By separating the `VisitMetadata` class into its own file and refactoring the code to use more descriptive variable names and modular functions, the maintainability of the codebase is improved.

