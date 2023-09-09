import { createContext, useState } from "react";
import PropTypes from "prop-types";
import { getUniqueNodeId } from "utils/genericHelper";
import { cloneDeep } from "lodash";

const initialFlowContext = {
  reactFlowInstance: null,
  setReactFlowInstance: () => {},
  deleteNode: () => {},
  deleteEdge: () => {},
  duplicateNode: () => {},
};

export const FlowContext = createContext(initialFlowContext);

export const ReactFlowContext = ({ children }) => {
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  /**
   * Deletes a node and all the edges connected to it.
   * @param {string} nodeId
   */
  const deleteNode = (nodeId) => {
    const nodes = reactFlowInstance.getNodes();
    const edges = reactFlowInstance.getEdges();

    const filteredNodes = nodes.filter((n) => n.id !== nodeId);
    const filteredEdges = edges.filter(
      (e) => e.source !== nodeId && e.target !== nodeId
    );

    reactFlowInstance.setNodes(filteredNodes);
    reactFlowInstance.setEdges(filteredEdges);
  };

  /**
   * Deletes an edge.
   * @param {string} edgeId
   */
  const deleteEdge = (edgeId) => {
    const edges = reactFlowInstance.getEdges();

    const filteredEdges = edges.filter((e) => e.id !== edgeId);

    reactFlowInstance.setEdges(filteredEdges);
  };

  /**
   * Creates a duplicate of an existing node.
   * @param {string} nodeId
   */
  const duplicateNode = (nodeId) => {
    const nodes = reactFlowInstance.getNodes();
    const originalNode = nodes.find((n) => n.id === nodeId);

    if (originalNode) {
      const newNodeId = getUniqueNodeId(originalNode.data, nodes);
      const clonedNode = cloneDeep(originalNode);

      const duplicatedNode = {
        ...clonedNode,
        id: newNodeId,
        position: {
          x: clonedNode.position.x + 400,
          y: clonedNode.position.y,
        },
        positionAbsolute: {
          x: clonedNode.positionAbsolute.x + 400,
          y: clonedNode.positionAbsolute.y,
        },
        data: {
          ...clonedNode.data,
          id: newNodeId,
        },
        selected: false,
      };

      // replace the ids of the connected edges
      for (const anchor of duplicatedNode.data.inputAnchors) {
        const connectedEdge = reactFlowInstance
          .getEdges()
          .find((e) => e.target === nodeId && e.targetHandle === anchor.id);

        if (connectedEdge) {
          connectedEdge.target = newNodeId;
          connectedEdge.targetHandle = anchor.id.replace(nodeId, newNodeId);
        }
      }

      reactFlowInstance.setNodes([...nodes, duplicatedNode]);
    }
  };

  return (
    <FlowContext.Provider
      value={{
        reactFlowInstance,
        setReactFlowInstance,
        deleteNode,
        deleteEdge,
        duplicateNode,
      }}
    >
      {children}
    </FlowContext.Provider>
  );
};

ReactFlowContext.propTypes = {
  children: PropTypes.any,
};

