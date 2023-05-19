import React, { useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { Box, AppBar, Toolbar } from '@mui/material';
import { useTheme } from '@mui/material/styles';

import ReactFlow, { Controls, Background, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
// import 'views/canvas/index.css';

import MarketplaceCanvasNode from './MarketplaceCanvasNode';
import MarketplaceCanvasHeader from './MarketplaceCanvasHeader';

// Define node and edge types
const nodeTypes = { customNode: MarketplaceCanvasNode };
const edgeTypes = { buttonedge: '' };

const MarketplaceCanvas = () => {
  // Get theme and navigation hooks
  const theme = useTheme();
  const navigate = useNavigate();

  // Get location and flow data
  const { state } = useLocation();
  const { flowData, name } = state;

  // Get nodes and edges state
  const [nodes, setNodes, onNodesChange] = useNodesState();
  const [edges, setEdges, onEdgesChange] = useEdgesState();

  // Create ref for reactflow wrapper
  const reactFlowWrapper = useRef(null);

  // Helper function to populate nodes and edges from flow data
  const populateNodesAndEdges = () => {
    if (flowData) {
      const initialFlow = JSON.parse(flowData);
      setNodes(initialFlow.nodes || []);
      setEdges(initialFlow.edges || []);
    }
  };

  // Initialize nodes and edges state from flow data. 
  useEffect(() => {
    populateNodesAndEdges();
  }, [flowData]);

  // Helper function to handle copying chatflow
  const onChatflowCopy = (flowData) => {
    const templateFlowData = JSON.stringify(flowData);
    navigate(`/canvas`, { state: { templateFlowData } });
  };

  return (
    <Box>
      {/*Header*/}
      <AppBar
        enableColorOnDark
        position="fixed"
        color="inherit"
        elevation={1}
        sx={{ bgcolor: theme.palette.background.default }}
      >
        <Toolbar>
          <MarketplaceCanvasHeader
            flowName={name}
            flowData={JSON.parse(flowData)}
            onChatflowCopy={(flowData) => onChatflowCopy(flowData)}
          />
        </Toolbar>
      </AppBar>
      {/*Main content*/}
      <Box sx={{ pt: '70px', height: '100vh', width: '100%' }}>
        <div className="reactflow-parent-wrapper">
          <div className="reactflow-wrapper" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodesDraggable={false}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              fitView
              minZoom={0.1}
            >
              {/*Controls*/}
              <Controls
                style={{
                  display: 'flex',
                  flexDirection: 'row',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                }}
              />
              {/*Background*/}
              <Background color="#aaa" gap={16} />
            </ReactFlow>
          </div>
        </div>
      </Box>
    </Box>
  );
};

export default MarketplaceCanvas;

