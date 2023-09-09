// To improve the maintainability of this codebase, I would suggest the following changes:

// 1. Extract the functionality for calculating the low discovery time of a vertex into a separate function. This will make the code more modular and easier to understand.

// 2. Rename the `dfsCallbacks` object to something more descriptive, such as `dfsCallbacksHandler`, to enhance readability.

// 3. Move the logic for checking if a vertex is an articulation point into a separate function. This will make the main `articulationPoints` function cleaner and easier to read.

// 4. Add comments to explain the purpose and functionality of each section of the code.

// Here is the refactored code:

// ```javascript
import depthFirstSearch from "../../CONSTANT/javascript_algorithms/depthFirstSearch";

/**
 * Helper class for visited vertex metadata.
 */
class VisitMetadata {
  constructor({ discoveryTime, lowDiscoveryTime }) {
    this.discoveryTime = discoveryTime;
    this.lowDiscoveryTime = lowDiscoveryTime;
    // We need this in order to check graph root node, whether it has two
    // disconnected children or not.
    this.independentChildrenCount = 0;
  }
}

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

  /**
   * Calculate the low discovery time of a vertex.
   *
   * @param {GraphVertex} vertex
   * @param {GraphVertex} parent
   * @returns {number}
   */
  function calculateLowDiscoveryTime(vertex, parent) {
    // Get minimum low discovery time from all neighbors.
    return vertex.getNeighbors()
      .filter((neighbor) => neighbor.getKey() !== parent.getKey())
      .reduce(
        (lowestDiscoveryTime, neighbor) => {
          const neighborLowTime = visitedSet[neighbor.getKey()].lowDiscoveryTime;
          return neighborLowTime < lowestDiscoveryTime ? neighborLowTime : lowestDiscoveryTime;
        },
        visitedSet[vertex.getKey()].lowDiscoveryTime,
      );
  }

  /**
   * Check if a vertex is an articulation point.
   *
   * @param {GraphVertex} vertex
   * @param {GraphVertex} parent
   */
  function checkArticulationPoint(vertex, parent) {
    if (parent === null) {
      // Don't do anything for the root vertex if it is already current (not previous one)
      return;
    }

    // Update the low time with the smallest time of adjacent vertices.
    const lowTime = calculateLowDiscoveryTime(vertex, parent);
    visitedSet[vertex.getKey()].lowDiscoveryTime = lowTime;

    // Detect whether parent vertex is an articulation point or not.
    if (parent === startVertex) {
      // Check that root vertex has at least two independent children.
      if (visitedSet[parent.getKey()].independentChildrenCount >= 2) {
        articulationPointsSet[parent.getKey()] = parent;
      }
    } else {
      // Get the low discovery time of the current vertex.
      const currentLowDiscoveryTime = visitedSet[vertex.getKey()].lowDiscoveryTime;

      // Compare the current vertex low discovery time with parent discovery time.
      // If the current vertex can't be reached via a path other than the parent,
      // then the parent vertex is an articulation point.
      const parentDiscoveryTime = visitedSet[parent.getKey()].discoveryTime;
      if (parentDiscoveryTime <= currentLowDiscoveryTime) {
        articulationPointsSet[parent.getKey()] = parent;
      }
    }
  }

  /**
   * Traverse the graph using depth-first search.
   *
   * @param {object} params
   * @param {GraphVertex} params.currentVertex
   * @param {GraphVertex} params.previousVertex
   */
  const dfsCallbacksHandler = {
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

      checkArticulationPoint(currentVertex, previousVertex);
    },

    allowTraversal: ({ nextVertex }) => {
      return !visitedSet[nextVertex.getKey()];
    },
  };

  // Do Depth First Search traversal over submitted graph.
  depthFirstSearch(graph, startVertex, dfsCallbacksHandler);

  return articulationPointsSet;
} 

