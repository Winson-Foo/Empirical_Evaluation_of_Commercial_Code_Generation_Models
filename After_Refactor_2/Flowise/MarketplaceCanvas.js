import React, { useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ReactFlow, { Background, Controls, useEdgesState, useNodesState } from 'reactflow';
import { AppBar, Box, Toolbar } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import 'reactflow/dist/style.css';
import 'views/canvas/index.css';
import MarketplaceCanvasHeader from './MarketplaceCanvasHeader';
import MarketplaceCanvasNode from './MarketplaceCanvasNode';

const nodeTypes = { customNode: MarketplaceCanvasNode };
const edgeTypes = { buttonedge: '' };

const MarketplaceCanvas = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { state } = useLocation();
  const { flowData, name } = state;

  const [nodes, setNodes, onNodesChange] = useNodesState();
  const [edges, setEdges, onEdgesChange] = useEdgesState();
  const reactFlowWrapper = useRef(null);

  useEffect(() => {
    initializeFlowData();
  }, [flowData]);

  const initializeFlowData = () => {
    if (flowData) {
      const initialFlow = JSON.parse(flowData);
      setNodes(initialFlow.nodes || []);
      setEdges(initialFlow.edges || []);
    }
  };

  const onChatflowCopy = (flowData) => {
    const templateFlowData = JSON.stringify(flowData);
    navigate(`/canvas`, { state: { templateFlowData } });
  };

  return (
    <>
      <Box>
        <AppBar
          enableColorOnDark
          position="fixed"
          color="inherit"
          elevation={1}
          sx={{
            bgcolor: theme.palette.background.default
          }}
        >
          <Toolbar>
            <MarketplaceCanvasHeader
              name={name}
              flowData={JSON.parse(flowData)}
              onChatflowCopy={onChatflowCopy}
            />
          </Toolbar>
        </AppBar>
        <Box sx={{ pt: '70px', height: '100vh', width: '100%' }}>
          <div className="reactflow-parent-wrapper">
            <div className="reactflow-wrapper" ref={reactFlowWrapper}>
              <ChatFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
              />
            </div>
          </div>
        </Box>
      </Box>
    </>
  );
};

const ChatFlow = ({ nodes, edges, onNodesChange, onEdgesChange }) => {
  return (
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
      <Controls
        style={{
          display: 'flex',
          flexDirection: 'row',
          left: '50%',
          transform: 'translate(-50%, -50%)'
        }}
      />
      <Background color="#aaa" gap={16} />
    </ReactFlow>
  );
};

export default MarketplaceCanvas;

