import React, { useContext, useEffect, useRef, useState } from 'react'
import PropTypes from 'prop-types'
import { Handle, Position, useUpdateNodeInternals } from 'reactflow'

import { useTheme, styled } from '@mui/material/styles'
import { Box, Tooltip, Typography } from '@mui/material'
import { tooltipClasses } from '@mui/material/Tooltip'

import { flowContext } from 'store/context/ReactFlowContext'
import { isValidConnection } from 'utils/genericHelper'
import { Dropdown } from 'ui-component/dropdown/Dropdown'

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 500
    }
})

const OutputHandle = ({ outputAnchor, data, position, reactFlowInstance, onSelectionChange }) => {
    const theme = useTheme()

    const handleConnection = (connection) => {
        return isValidConnection(connection, reactFlowInstance)
    }

    const handleSelectionChange = (newValue) => {
        data.outputs[outputAnchor.name] = newValue
        onSelectionChange(newValue)
    }

    const optionType = outputAnchor.options?.find((opt) => opt.name === data.outputs?.[outputAnchor.name])?.type ?? outputAnchor.type
    const optionValue = data.outputs[outputAnchor.name] ?? outputAnchor.default ?? 'choose an option'
    const optionId = outputAnchor.options?.find((opt) => opt.name === data.outputs?.[outputAnchor.name])?.id ?? ''

    return (
        <div>
            <CustomWidthTooltip placement='right' title={optionType}>
                <Handle
                    type='source'
                    position={Position.Right}
                    key={outputAnchor.id}
                    id={optionId}
                    isValidConnection={handleConnection}
                    style={{
                        height: 10,
                        width: 10,
                        backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                        top: position
                    }}
                />
            </CustomWidthTooltip>
            <Box sx={{ p: 2, textAlign: 'end' }}>
                <Dropdown
                    name={outputAnchor.name}
                    options={outputAnchor.options}
                    value={optionValue}
                    disabled={outputAnchor.type === 'options' && !outputAnchor.options}
                    disableClearable
                    onSelect={handleSelectionChange}
                />
            </Box>
        </div>
    )
}

OutputHandle.propTypes = {
    outputAnchor: PropTypes.object,
    data: PropTypes.object,
    position: PropTypes.number,
    reactFlowInstance: PropTypes.object,
    onSelectionChange: PropTypes.func,
}

const NodeOutputHandler = ({ outputAnchor, data }) => {
    const [position, setPosition] = useState(0)
    const [dropdownValue, setDropdownValue] = useState(null)
    const { reactFlowInstance } = useContext(flowContext)
    const updateNodeInternals = useUpdateNodeInternals()
    const ref = useRef(null)

    useEffect(() => {
        if (ref.current) {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2)
            updateNodeInternals(data.id)
        }
    }, [data.id, updateNodeInternals])

    const handleSelectionChange = (newValue) => {
        setTimeout(() => {
            setDropdownValue(newValue)
            updateNodeInternals(data.id)
        }, 0)
    }

    return (
        <div ref={ref}>
            {outputAnchor.type !== 'options' && !outputAnchor.options && (
                <>
                    <OutputHandle
                        outputAnchor={outputAnchor}
                        data={data}
                        position={position}
                        reactFlowInstance={reactFlowInstance}
                        onSelectionChange={handleSelectionChange}
                    />
                    <Box sx={{ p: 2, textAlign: 'end' }}>
                        <Typography>{outputAnchor.label}</Typography>
                    </Box>
                </>
            )}
            {outputAnchor.type === 'options' && outputAnchor.options && outputAnchor.options.length > 0 && (
                <OutputHandle
                    outputAnchor={outputAnchor}
                    data={data}
                    position={position}
                    reactFlowInstance={reactFlowInstance}
                    onSelectionChange={handleSelectionChange}
                />
            )}
        </div>
    )
}

NodeOutputHandler.defaultProps = {
    outputAnchor: {},
    data: {},
}

NodeOutputHandler.propTypes = {
    outputAnchor: PropTypes.object.isRequired,
    data: PropTypes.object.isRequired,
}

export default NodeOutputHandler

