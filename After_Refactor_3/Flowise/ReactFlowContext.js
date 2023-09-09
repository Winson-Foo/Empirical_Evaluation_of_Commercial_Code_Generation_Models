import React, { createContext, useReducer } from 'react';
import PropTypes from 'prop-types';
import { getUniqueNodeId } from 'utils/genericHelper';
import { cloneDeep } from 'lodash';

export const FlowContext = createContext();

const initialState = {
  reactFlowInstance: null,
  nodes: [],
  edges: [],
};

const reducer = (state, action) => {
  switch (action.type) {
    case 'setInstance':
      return { ...state, reactFlowInstance: action.payload };
    case 'setNodes':
      return { ...state, nodes: action.payload };
    case 'setEdges':
      return { ...state, edges: action.payload };
    case 'deleteNode':
      return {
        ...state,
        nodes: state.nodes.filter((node) => node.id !== action.payload.id),
        edges: state.edges.filter(
          (edge) => edge.source !== action.payload.id && edge.target !== action.payload.id
        ),
      };
    case 'deleteEdge':
      return {
        ...state,
        edges: state.edges.filter((edge) => edge.id !== action.payload.id),
      };
    case 'duplicateNode':
      const { nodes, edges } = state;
      const { id } = action.payload;
      const originalNode = nodes.find((n) => n.id === id);
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

        const dataKeys = ['inputParams', 'inputAnchors', 'outputAnchors'];

        for (const key of dataKeys) {
          for (const item of duplicatedNode.data[key]) {
            if (item.id) {
              item.id = item.id.replace(id, newNodeId);
            }
          }
        }

        return {
          ...state,
          nodes: [...nodes, duplicatedNode],
        };
      }
      break;
    default:
      throw new Error(`Invalid action type: ${action.type}`);
  }
};

export const ReactFlowContext = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);

  const setReactFlowInstance = (instance) => {
    dispatch({ type: 'setInstance', payload: instance });
  };

  const setNodes = (nodes) => {
    dispatch({ type: 'setNodes', payload: nodes });
  };

  const setEdges = (edges) => {
    dispatch({ type: 'setEdges', payload: edges });
  };

  const deleteNode = (id) => {
    dispatch({ type: 'deleteNode', payload: { id } });
  };

  const deleteEdge = (id) => {
    dispatch({ type: 'deleteEdge', payload: { id } });
  };

  const duplicateNode = (id) => {
    dispatch({ type: 'duplicateNode', payload: { id } });
  };

  const contextValue = {
    ...state,
    setReactFlowInstance,
    setNodes,
    setEdges,
    deleteNode,
    deleteEdge,
    duplicateNode,
  };

  return (
    <FlowContext.Provider value={contextValue}>{children}</FlowContext.Provider>
  );
};

ReactFlowContext.propTypes = {
  children: PropTypes.any,
};


