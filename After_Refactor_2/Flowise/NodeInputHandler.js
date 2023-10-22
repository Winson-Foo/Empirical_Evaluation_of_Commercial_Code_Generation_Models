import PropTypes from 'prop-types'
import { Handle, Position, useUpdateNodeInternals } from 'reactflow'
import { useEffect, useRef, useState, useContext } from 'react'
import { useSelector } from 'react-redux'

// material-ui
import { useTheme, styled } from '@mui/material/styles'
import { Box, Typography, Tooltip, IconButton } from '@mui/material'
import { tooltipClasses } from '@mui/material/Tooltip'
import { IconArrowsMaximize } from '@tabler/icons'

// project import
import { Dropdown } from 'ui-component/dropdown/Dropdown'
import { Input } from 'ui-component/input/Input'
import { FileInput } from 'ui-component/file/FileInput'
import { SwitchInput } from 'ui-component/switch/SwitchInput'
import { JsonEditorInput } from 'ui-component/json/JsonEditorInput'
import { isValidConnection, getAvailableNodesForVariable } from 'utils/genericHelper'
import { flowContext } from 'store/context/ReactFlowContext'

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 500
    }
})

const LABEL_REQUIRED = <span style={{ color: 'red' }}>&nbsp;*</span>
const DIALOG_CONFIRM = 'Save'
const DIALOG_CANCEL = 'Cancel'
const EXPAND_ICON_TITLE = 'Expand'

const NodeInputHandler = ({ inputAnchor, inputParam, data, disabled = false, isAdditionalParams = false }) => {
    const theme = useTheme()
    const customization = useSelector((state) => state.customization)
    const ref = useRef(null)
    const { reactFlowInstance } = useContext(flowContext)
    const updateNodeInternals = useUpdateNodeInternals()
    const [position, setPosition] = useState(0)
    const [showExpandDialog, setShowExpandDialog] = useState(false)
    const [expandDialogProps, setExpandDialogProps] = useState({})

    const onExpandDialogClicked = () => {
        const newValue = data.inputs[inputParam.name] ?? inputParam.default ?? ''

        const dialogProp = {
            value: newValue,
            inputParam,
            disabled,
            confirmButtonName: DIALOG_CONFIRM,
            cancelButtonName: DIALOG_CANCEL
        }

        if (!disabled) {
            const nodes = reactFlowInstance.getNodes()
            const edges = reactFlowInstance.getEdges()
            const nodesForVariable = inputParam.acceptVariable ? getAvailableNodesForVariable(nodes, edges, data.id, inputParam.id) : []
            dialogProp.availableNodesForVariable = nodesForVariable
        }

        setExpandDialogProps(dialogProp)
        setShowExpandDialog(true)
    }

    const onExpandDialogSave = (newValue, inputParamName) => {
        setShowExpandDialog(false)
        data.inputs[inputParamName] = newValue
    }

    const renderInput = () => {
        if (inputParam.type === 'file') {
            return (
                <FileInput
                    disabled={disabled}
                    fileType={inputParam.fileType || '*'}
                    onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                    value={data.inputs[inputParam.name] ?? inputParam.default ?? 'Choose a file to upload'}
                />
            )
        }

        if (inputParam.type === 'boolean') {
            return (
                <SwitchInput
                    disabled={disabled}
                    onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                    value={data.inputs[inputParam.name] ?? inputParam.default ?? false}
                />
            )
        }

        if (['string', 'password', 'number'].includes(inputParam.type)) {
            return (
                <Input
                    disabled={disabled}
                    inputParam={inputParam}
                    onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                    value={data.inputs[inputParam.name] ?? inputParam.default ?? ''}
                    showDialog={showExpandDialog}
                    dialogProps={expandDialogProps}
                    onDialogCancel={() => setShowExpandDialog(false)}
                    onDialogConfirm={(newValue, inputParamName) => onExpandDialogSave(newValue, inputParamName)}
                />
            )
        }

        if (inputParam.type === 'json') {
            return (
                <JsonEditorInput
                    disabled={disabled}
                    onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                    value={data.inputs[inputParam.name] ?? inputParam.default ?? ''}
                    isDarkMode={customization.isDarkMode}
                />
            )
        }

        if (inputParam.type === 'options') {
            return (
                <Dropdown
                    disabled={disabled}
                    name={inputParam.name}
                    options={inputParam.options}
                    onSelect={(newValue) => (data.inputs[inputParam.name] = newValue)}
                    value={data.inputs[inputParam.name] ?? inputParam.default ?? 'chose an option'}
                />
            )
        }

        return null
    }

    useEffect(() => {
        if (ref.current && ref.current.offsetTop && ref.current.clientHeight) {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2)
            updateNodeInternals(data.id)
        }
    }, [data.id, ref, updateNodeInternals])

    useEffect(() => {
        updateNodeInternals(data.id)
    }, [data.id, position, updateNodeInternals])

    return (
        <div ref={ref}>
            {inputAnchor && (
                <>
                    <CustomWidthTooltip placement='left' title={inputAnchor.type}>
                        <Handle
                            type='target'
                            position={Position.Left}
                            key={inputAnchor.id}
                            id={inputAnchor.id}
                            isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                            style={{
                                height: 10,
                                width: 10,
                                backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                                top: position
                            }}
                        />
                    </CustomWidthTooltip>
                    <Box sx={{ p: 2 }}>
                        <Typography>
                            {inputAnchor.label}
                            {!inputAnchor.optional && LABEL_REQUIRED}
                        </Typography>
                    </Box>
                </>
            )}

            {((inputParam && !inputParam.additionalParams) || isAdditionalParams) && (
                <>
                    {inputParam.acceptVariable && (
                        <CustomWidthTooltip placement='left' title={inputParam.type}>
                            <Handle
                                type='target'
                                position={Position.Left}
                                key={inputParam.id}
                                id={inputParam.id}
                                isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                                style={{
                                    height: 10,
                                    width: 10,
                                    backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                                    top: position
                                }}
                            />
                        </CustomWidthTooltip>
                    )}
                    <Box sx={{ p: 2 }}>
                        <div style={{ display: 'flex', flexDirection: 'row' }}>
                            <Typography>
                                {inputParam.label}
                                {!inputParam.optional && LABEL_REQUIRED}
                            </Typography>
                            <div style={{ flexGrow: 1 }}></div>
                            {inputParam.type === 'string' && inputParam.rows && (
                                <IconButton
                                    size='small'
                                    sx={{
                                        height: 25,
                                        width: 25
                                    }}
                                    title={EXPAND_ICON_TITLE}
                                    color='primary'
                                    onClick={() => onExpandDialogClicked()}
                                >
                                    <IconArrowsMaximize />
                                </IconButton>
                            )}
                        </div>
                        {renderInput()}
                    </Box>
                </>
            )}
        </div>
    )
}

NodeInputHandler.propTypes = {
    inputAnchor: PropTypes.object,
    inputParam: PropTypes.object,
    data: PropTypes.object,
    disabled: PropTypes.bool,
    isAdditionalParams: PropTypes.bool
}

export default NodeInputHandler

