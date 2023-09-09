import PropTypes from 'prop-types'
import { Handle, Position, useUpdateNodeInternals } from 'reactflow'
import { useEffect, useRef, useState, useContext } from 'react'

// material-ui
import { useTheme, styled } from '@mui/material/styles'
import { Box, Typography, Tooltip } from '@mui/material'
import { tooltipClasses } from '@mui/material/Tooltip'
import { flowContext } from 'store/context/ReactFlowContext'
import { isValidConnection } from 'utils/genericHelper'
import { Dropdown } from 'ui-component/dropdown/Dropdown'

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 500
  }
})

const NodeOutputHandler = ({ outputAnchor, data, disabled = false }) => {
  const theme = useTheme()
  const outputRef = useRef(null)
  const updateNodeInternals = useUpdateNodeInternals()
  const { reactFlowInstance } = useContext(flowContext)
  const [handleTop, setHandleTop] = useState(0)
  const [dropdownValue, setDropdownValue] = useState(null)
  const { type, id, name, label, options, default: defaultOption } = outputAnchor;

  useEffect(() => {
    if (outputRef.current) {
      setHandleTop(outputRef.current.offsetTop + outputRef.current.clientHeight / 2)
      updateNodeInternals(data.id)
    }
  }, [data.id, updateNodeInternals])

  useEffect(() => {
    updateNodeInternals(data.id)
  }, [data.id, handleTop, updateNodeInternals])

  useEffect(() => {
    if (dropdownValue) {
      updateNodeInternals(data.id)
    }
  }, [data.id, dropdownValue, updateNodeInternals])

  const renderHandle = (type, id, isValidConnection, backgroundColor, top) => (
    <CustomWidthTooltip placement='right' title={type}>
      <Handle
        type='source'
        position={Position.Right}
        key={id}
        id={id}
        isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
        style={{
          height: 10,
          width: 10,
          backgroundColor: backgroundColor,
          top: top
        }}
      />
    </CustomWidthTooltip>
  );

  return (
    <div ref={outputRef}>
      {type !== 'options' && !options && (
        <>
          {renderHandle(type, id, isValidConnection, data.selected ? theme.palette.primary.main : theme.palette.text.secondary, handleTop)}
          <Box sx={{ p: 2, textAlign: 'end' }}>
            <Typography>{label}</Typography>
          </Box>
        </>
      )}
      {type === 'options' && options && options.length > 0 && (
        <>
          {renderHandle(options.find((opt) => opt.name === data.outputs?.[name])?.type ?? type, options.find((opt) => opt.name === data.outputs?.[name])?.id ?? '', isValidConnection, data.selected ? theme.palette.primary.main : theme.palette.text.secondary, handleTop)}
          <Box sx={{ p: 2, textAlign: 'end' }}>
            <Dropdown
              disabled={disabled}
              disableClearable={true}
              name={name}
              options={options}
              onSelect={(newValue) => {
                setDropdownValue(newValue)
                data.outputs[name] = newValue
              }}
              value={data.outputs[name] ?? defaultOption ?? 'choose an option'}
            />
          </Box>
        </>
      )}
    </div>
  )
}

NodeOutputHandler.propTypes = {
  outputAnchor: PropTypes.object,
  data: PropTypes.object,
  disabled: PropTypes.bool
}

export default NodeOutputHandler