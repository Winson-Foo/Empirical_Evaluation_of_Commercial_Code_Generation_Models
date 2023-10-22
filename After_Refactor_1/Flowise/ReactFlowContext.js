import { createContext, useState } from 'react';
import PropTypes from 'prop-types';
import { getUniqueNodeId } from 'utils/genericHelper';
import { cloneDeep } from 'lodash';

const initialValue = {
  reactFlowInstance: null,
  setReactFlowInstance: () => {},
  duplicateNode: () => {},
  deleteNode: () => {},
  deleteEdge: () => {},
};

export const flowContext = createContext(initialValue);

export const ReactFlowContext = ({ children }) => {
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const deleteNode = (nodeId) => {
    const { setEdges, getEdges, setNodes, getNodes } = reactFlowInstance;
    deleteConnectedInput(nodeId, 'node');
    setNodes(getNodes().filter(node => node.id !== nodeId));
    setEdges(getEdges().filter(edge => edge.source !== nodeId && edge.target !== nodeId));
  };

  const deleteEdge = (edgeId) => {
    const { setEdges, getEdges } = reactFlowInstance;
    deleteConnectedInput(edgeId, 'edge');
    setEdges(getEdges().filter(edge => edge.id !== edgeId));
  };

  const deleteConnectedInput = (id, type) => {
    const { setNodes, getNodes, getEdges } = reactFlowInstance;
    const connectedEdges =
      type === 'node'
        ? getEdges().filter(edge => edge.source === id)
        : getEdges().filter(edge => edge.id === id);
    const targetInputKey = 'targetHandle';
    const instanceKey = 'data.instance';
    for (const edge of connectedEdges) {
      const targetNodeId = edge.target;
      const sourceNodeId = edge.source;
      const targetInput = edge[targetInputKey].split('-')[2];
      setNodes((nodes) => nodes.map((node) => {
        if (node.id === targetNodeId) {
          let value;
          const inputAnchor = node.data.inputAnchors.find(anchor => anchor.name === targetInput);
          const inputParam = node.data.inputParams.find(param => param.name === targetInput);
          const inputValue = node.data.inputs[targetInput];
          const listKey = 'list';
          const isList = inputAnchor && inputAnchor[listKey];
          const acceptVariableKey = 'acceptVariable';
          const acceptsVariable = inputParam && inputParam[acceptVariableKey];
          if (isList) {
            value = inputValue.filter(item => !item.includes(sourceNodeId));
          } else if (acceptsVariable) {
            const pattern = `{{${sourceNodeId}.${instanceKey}}}`;
            value = inputValue.replace(pattern, '') || '';
          } else {
            value = '';
          }
          node.data = {
            ...node.data,
            inputs: {
              ...node.data.inputs,
              [targetInput]: value,
            },
          };
        }
        return node;
      }));
    }
  };

  const duplicateNode = (id) => {
    const { setNodes, getNodes } = reactFlowInstance;
    const nodes = getNodes();
    const originalNode = nodes.find(node => node.id === id);
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

      setNodes([...nodes, duplicatedNode]);
    }
  };

  return (
    <flowContext.Provider
      value={{
        reactFlowInstance,
        setReactFlowInstance,
        deleteNode,
        deleteEdge,
        duplicateNode,
      }}
    >
      {children}
    </flowContext.Provider>
  );
};

ReactFlowContext.propTypes = {
  children: PropTypes.any,
};

