// ButtonEdge.js
import { getBezierPath, EdgeText } from 'reactflow'
import PropTypes from 'prop-types'
import { useDispatch } from 'react-redux'
import { useContext } from 'react'
import { SET_DIRTY } from 'store/actions'
import { flowContext } from 'store/context/ReactFlowContext'

import './index.css'

const foreignObjectSize = 40

const ButtonEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, label, markerEnd }) => {
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

  const handleClick = (evt, id) => {
    evt.stopPropagation()
    deleteEdge(id)
    dispatch({ type: SET_DIRTY })
  }

  return (
    <>
      <path id={id} style={style} className='react-flow__edge-path' d={edgePath} markerEnd={markerEnd} />
      {label && (
        <ButtonEdgeLabel sourceX={sourceX} sourceY={sourceY} label={label} />
      )}
      <ButtonEdgeButton center={{ x: edgeCenterX, y: edgeCenterY }} id={id} onClick={handleClick} />
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
  label: PropTypes.string,
  markerEnd: PropTypes.any
}

const ButtonEdgeLabel = ({ sourceX, sourceY, label }) => {
  return (
    <EdgeText
      x={sourceX + 10}
      y={sourceY + 10}
      label={label}
      labelStyle={{ fill: 'black' }}
      labelBgStyle={{ fill: 'transparent' }}
      labelBgPadding={[2, 4]}
      labelBgBorderRadius={2}
    />
  )
}

ButtonEdgeLabel.propTypes = {
  sourceX: PropTypes.number,
  sourceY: PropTypes.number,
  label: PropTypes.string
}

const ButtonEdgeButton = ({ center, id, onClick }) => {
  const { x, y } = center

  return (
    <foreignObject
      width={foreignObjectSize}
      height={foreignObjectSize}
      x={x - foreignObjectSize / 2}
      y={y - foreignObjectSize / 2}
      className='edgebutton-foreignobject'
      requiredExtensions='http://www.w3.org/1999/xhtml'
    >
      <div>
        <button className='edgebutton' onClick={(event) => onClick(event, id)}>
          �
        </button>
      </div>
    </foreignObject>
  )
}

ButtonEdgeButton.propTypes = {
  center: PropTypes.object,
  id: PropTypes.string,
  onClick: PropTypes.func
}

export default ButtonEdge

