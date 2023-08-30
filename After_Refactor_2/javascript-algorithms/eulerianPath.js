// To improve the maintainability of the codebase, I would suggest the following refactored code:

import graphBridges from "../../CONSTANT/javascript_algorithms/graphBridges";

/**
 * Fleury's algorithm of finding Eulerian Path (visit all graph edges exactly once).
 *
 * @param {Graph} graph
 * @return {GraphVertex[]}
 */
export default function eulerianPath(graph) {
  const eulerianPathVertices = [];
  const evenRankVertices = new Set();
  const oddRankVertices = new Set();
  const notVisitedEdges = new Set(graph.getAllEdges());

  graph.getAllVertices().forEach((vertex) => {
    if (vertex.getDegree() % 2 === 0) {
      evenRankVertices.add(vertex);
    } else {
      oddRankVertices.add(vertex);
    }
  });

  const isCircuit = oddRankVertices.size === 0;

  if (!isCircuit && oddRankVertices.size !== 2) {
    throw new Error('Eulerian path must contain two odd-ranked vertices');
  }

  let startVertex = null;

  if (isCircuit) {
    const evenVertex = evenRankVertices.values().next().value;
    startVertex = evenVertex;
  } else {
    const oddVertex = oddRankVertices.values().next().value;
    startVertex = oddVertex;
  }

  let currentVertex = startVertex;
  while (notVisitedEdges.size > 0) {
    eulerianPathVertices.push(currentVertex);

    const bridges = graphBridges(graph);
    let edgeToDelete = null;

    if (currentVertex.getEdges().length === 1) {
      [edgeToDelete] = currentVertex.getEdges();
    } else {
      [edgeToDelete] = currentVertex.getEdges().filter((edge) => !bridges.has(edge));
    }

    if (currentVertex === edgeToDelete.startVertex) {
      currentVertex = edgeToDelete.endVertex;
    } else {
      currentVertex = edgeToDelete.startVertex;
    }

    notVisitedEdges.delete(edgeToDelete);

    if (notVisitedEdges.size === 0) {
      eulerianPathVertices.push(currentVertex);
    }

    graph.deleteEdge(edgeToDelete);
  }

  return eulerianPathVertices;
}

