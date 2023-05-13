import { getBezierPath, EdgeText } from 'reactflow'
import PropTypes from 'prop-types'
import { useDispatch } from 'react-redux'
import { useContext } from 'react'
import { SET_DIRTY } from 'store/actions'
import { flowContext } from 'store/context/ReactFlowContext'

import './index.css'

const ButtonEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, data, markerEnd }) => {
  const [edgePath, edgeCenterX, edgeCenterY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition
  })

  const { deleteEdge } = useContext(flowContext)
  const dispatch = useDispatch()

  const handleEdgeDelete = (evt, id) => {
    evt.stopPropagation()
    deleteEdge(id)
    dispatch({ type: SET_DIRTY })
  }

  return (
    <>
      <path id={id} style={style} className='react-flow__edge-path' d={edgePath} markerEnd={markerEnd} />
      {data?.label && (
        <EdgeText
          x={sourceX + 10}
          y={sourceY + 10}
          label={data.label}
          labelStyle={{ fill: 'black' }}
          labelBgStyle={{ fill: 'transparent' }}
          labelBgPadding='2 4'
          labelBgBorderRadius={2}
        />
      )}
      <foreignObject
        width='40'
        height='40'
        x={edgeCenterX - 20}
        y={edgeCenterY - 20}
        className='edgebutton-foreignobject'
        requiredExtensions='http://www.w3.org/1999/xhtml'
      >
        <div>
          <button className='edgebutton' onClick={(evt) => handleEdgeDelete(evt, id)}>
            �
          </button>
        </div>
      </foreignObject>
    </>
  )
}

ButtonEdge.propTypes = {
  id: PropTypes.string,
  sourceX: PropTypes.number,
  sourceY: PropTypes.number,
  targetX: PropTypes.number,
  targetY: PropTypes.number,
  sourcePosition: PropTypes.any,
  targetPosition: PropTypes.any,
  style: PropTypes.object,
  data: PropTypes.object,
  markerEnd: PropTypes.any
}

ButtonEdge.defaultProps = {
  style: {},
  data: {},
  markerEnd: null
}

export default ButtonEdge

