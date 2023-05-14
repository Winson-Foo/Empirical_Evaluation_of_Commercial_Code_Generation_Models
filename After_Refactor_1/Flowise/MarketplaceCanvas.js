// MarketplaceCanvas.jsx

import React, { useEffect, useRef } from 'react'
import PropTypes from 'prop-types'
import { useLocation, useNavigate } from 'react-router-dom'
import { Toolbar, Box, AppBar } from '@mui/material'
import { useTheme } from '@mui/material/styles'
import ReactFlow, { Controls, Background, useNodesState, useEdgesState } from 'reactflow'

import MarketplaceCanvasNode from './MarketplaceCanvasNode'
import MarketplaceCanvasHeader from './MarketplaceCanvasHeader'

import './MarketplaceCanvas.css'

const nodeTypes = { customNode: MarketplaceCanvasNode }
const edgeTypes = { buttonedge: '' }

const MarketplaceCanvas = ({ flowData: initialFlow, name: flowName }) => {
  const theme = useTheme()
  const navigate = useNavigate()
  const { state } = useLocation()
  const reactFlowWrapper = useRef(null)

  const [nodes, setNodes, onNodesChange] = useNodesState()
  const [edges, setEdges, onEdgesChange] = useEdgesState()

  useEffect(() => {
    if (state?.flowData) {
      const initialNodes = state.flowData.nodes || []
      const initialEdges = state.flowData.edges || []
      setNodes(initialNodes)
      setEdges(initialEdges)
    } else if (initialFlow) {
      const initialNodes = initialFlow.nodes || []
      const initialEdges = initialFlow.edges || []
      setNodes(initialNodes)
      setEdges(initialEdges)
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialFlow, state])

  const onChatflowCopy = (flowData) => {
    const templateFlowData = JSON.stringify(flowData)
    navigate('/canvas', { state: { templateFlowData } })
  }

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
              flowName={flowName}
              flowData={{ nodes, edges }}
              onChatflowCopy={onChatflowCopy}
            />
          </Toolbar>
        </AppBar>
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
            </div>
          </div>
        </Box>
      </Box>
    </>
  )
}

MarketplaceCanvas.propTypes = {
  flowData: PropTypes.shape({
    nodes: PropTypes.arrayOf(PropTypes.object),
    edges: PropTypes.arrayOf(PropTypes.object)
  }),
  name: PropTypes.string.isRequired
}

MarketplaceCanvas.defaultProps = {
  flowData: null
}

export default MarketplaceCanvas

