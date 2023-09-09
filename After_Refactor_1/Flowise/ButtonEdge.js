import { getBezierPath, EdgeText } from 'reactflow';
import PropTypes from 'prop-types';
import { useDispatch } from 'react-redux';
import { useContext } from 'react';
import { SET_DIRTY } from 'store/actions';
import { flowContext } from 'store/context/ReactFlowContext';
import './index.css';

// Button size for foreignObject
const foreignObjectSize = 40;

// Handler for deleting edges
const onEdgeClick = (evt, id, deleteEdge, dispatch) => {
  evt.stopPropagation(); // prevent the click event from propagating to the node
  deleteEdge(id); // delete the edge from the ReactFlow component
  dispatch({ type: SET_DIRTY }); // set the dirty flag in the Redux store
};

const ButtonEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd
}) => {
  // Get edge path and center
  const [edgePath, edgeCenterX, edgeCenterY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition
  });

  // Get deleteEdge and dispatch functions from ReactFlowContext and Redux
  const { deleteEdge } = useContext(flowContext);
  const dispatch = useDispatch();

  return (
    // Render the edge path, label and delete button
    <>
      <path
        id={id}
        style={style}
        className='react-flow__edge-path'
        d={edgePath}
        markerEnd={markerEnd}
      />
      {data && data.label && (
        <EdgeText
          x={sourceX + 10}
          y={sourceY + 10}
          label={data.label}
          labelStyle={{ fill: 'black' }}
          labelBgStyle={{ fill: 'transparent' }}
          labelBgPadding={[2, 4]}
          labelBgBorderRadius={2}
        />
      )}
      <foreignObject
        width={foreignObjectSize}
        height={foreignObjectSize}
        x={edgeCenterX - foreignObjectSize / 2}
        y={edgeCenterY - foreignObjectSize / 2}
        className='edgebutton-foreignobject'
        requiredExtensions='http://www.w3.org/1999/xhtml'
      >
        <div>
          <button
            className='edgebutton'
            onClick={(evt) => onEdgeClick(evt, id, deleteEdge, dispatch)}
          >
            �
          </button>
        </div>
      </foreignObject>
    </>
  );
};

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
};

export default ButtonEdge;