import PropTypes from 'prop-types';
import { Position, useUpdateNodeInternals } from 'reactflow';
import { useEffect, useRef, useState, useContext } from 'react';
import { useSelector } from 'react-redux';

// material-ui
import { useTheme, styled } from '@mui/material/styles';
import { Box, Typography, Tooltip, IconButton } from '@mui/material';
import { tooltipClasses } from '@mui/material/Tooltip';
import { IconArrowsMaximize } from '@tabler/icons';

// project import
import { Dropdown } from 'ui-component/dropdown/Dropdown';
import { Input } from 'ui-component/input/Input';
import { FileInput } from 'ui-component/file/FileInput';
import { SwitchInput } from 'ui-component/switch/SwitchInput';
import { JsonEditorInput } from 'ui-component/json/JsonEditor';
import { isValidConnection, getAvailableNodesForVariable } from 'utils/genericHelper';
import { flowContext } from 'store/context/ReactFlowContext';

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 500
    }
});

function HandleInputTarget({ id, isValidConnection, data, position }) {
    const { reactFlowInstance } = useContext(flowContext)
    const theme = useTheme()
    return (
        <CustomWidthTooltip placement='left' title={id}>
            <Handle
                type='target'
                position={Position.Left}
                key={id}
                id={id}
                isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                style={{
                    height: 10,
                    width: 10,
                    backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                    top: position,
                }}
            />
        </CustomWidthTooltip>
    );
}

HandleInputTarget.propTypes = {
    id: PropTypes.string.isRequired,
    position: PropTypes.number.isRequired,
    data: PropTypes.object.isRequired,
    isValidConnection: PropTypes.func.isRequired
}

function ExpandableInput({ inputParam, data }) {
    const [showExpandDialog, setShowExpandDialog] = useState(false);
    const [expandDialogProps, setExpandDialogProps] = useState({});
    const [position, setPosition] = useState(0);
    const theme = useTheme()
    const ref = useRef(null);
    const updateNodeInternals = useUpdateNodeInternals();
    const { reactFlowInstance } = useContext(flowContext);
    const customization = useSelector((state) => state.customization);

    const onExpandDialogClicked = (value, inputParam) => {
        const dialogProp = {
            value,
            inputParam,
            disabled: inputParam.disabled ?? false,
            confirmButtonName: 'Save',
            cancelButtonName: 'Cancel',
        };

        if (!inputParam.disabled) {
            const nodes = reactFlowInstance.getNodes();
            const edges = reactFlowInstance.getEdges();
            const nodesForVariable = inputParam.acceptVariable ? getAvailableNodesForVariable(nodes, edges, data.id, inputParam.id) : [];
            dialogProp.availableNodesForVariable = nodesForVariable;
        }
        setExpandDialogProps(dialogProp);
        setShowExpandDialog(true);
    };

    const onExpandDialogSave = (newValue, inputParamName) => {
        setShowExpandDialog(false);
        data.inputs[inputParamName] = newValue;
    };

    useEffect(() => {
        if (ref.current && ref.current.offsetTop && ref.current.clientHeight) {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2);
            updateNodeInternals(data.id);
        }
    }, [data.id, ref, updateNodeInternals]);

    useEffect(() => {
        updateNodeInternals(data.id);
    }, [data.id, position, updateNodeInternals]);

    return (
        <>
            <Box sx={{ p: 2 }}>
                <div style={{ display: 'flex', flexDirection: 'row' }}>
                    <Typography>
                        {inputParam.label}
                        {!inputParam.optional && <span style={{ color: 'red' }}>&nbsp;*</span>}
                    </Typography>
                    <div style={{ flexGrow: 1 }}></div>
                    {inputParam.type === 'string' && inputParam.rows && (
                        <IconButton
                            size='small'
                            sx={{
                                height: 25,
                                width: 25,
                            }}
                            title='Expand'
                            color='primary'
                            onClick={() => onExpandDialogClicked(data.inputs[inputParam.name] ?? inputParam.default ?? '', inputParam)}
                        >
                            <IconArrowsMaximize />
                        </IconButton>
                    )}
                </div>
                {inputParam.type === 'file' && (
                    <FileInput
                        disabled={inputParam.disabled ?? false}
                        fileType={inputParam.fileType ?? '*'}
                        onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                        value={data.inputs[inputParam.name] ?? inputParam.default ?? 'Choose a file to upload'}
                    />
                )}
                {inputParam.type === 'boolean' && (
                    <SwitchInput
                        disabled={inputParam.disabled ?? false}
                        onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                        value={data.inputs[inputParam.name] ?? inputParam.default ?? false}
                    />
                )}
                {(inputParam.type === 'string' || inputParam.type === 'password' || inputParam.type === 'number') && (
                    <Input
                        disabled={inputParam.disabled ?? false}
                        inputParam={inputParam}
                        onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                        value={data.inputs[inputParam.name] ?? inputParam.default ?? ''}
                        showDialog={showExpandDialog}
                        dialogProps={expandDialogProps}
                        onDialogCancel={() => setShowExpandDialog(false)}
                        onDialogConfirm={(newValue, inputParamName) => onExpandDialogSave(newValue, inputParamName)}
                    />
                )}
                {inputParam.type === 'json' && (
                    <JsonEditorInput
                        disabled={inputParam.disabled ?? false}
                        onChange={(newValue) => (data.inputs[inputParam.name] = newValue)}
                        value={data.inputs[inputParam.name] ?? inputParam.default ?? ''}
                        isDarkMode={customization.isDarkMode}
                    />
                )}
                {inputParam.type === 'options' && (
                    <Dropdown
                        disabled={inputParam.disabled ?? false}
                        name={inputParam.name}
                        options={inputParam.options}
                        onSelect={(newValue) => (data.inputs[inputParam.name] = newValue)}
                        value={data.inputs[inputParam.name] ?? inputParam.default ?? 'chose an option'}
                    />
                )}
            </Box>
        </>
    );
}

ExpandableInput.propTypes = {
    inputParam: PropTypes.object.isRequired,
    data: PropTypes.object.isRequired
}

function NodeInputHandler({ inputAnchor, inputParam, data, disabled = false, isAdditionalParams = false }) {
    const theme = useTheme()
    const ref = useRef(null)
    const { reactFlowInstance } = useContext(flowContext)
    const updateNodeInternals = useUpdateNodeInternals()
    const [position, setPosition] = useState(0)

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
                    <HandleInputTarget id={inputAnchor.id} position={position} data={data} isValidConnection={isValidConnection} />
                    <Box sx={{ p: 2 }}>
                        <Typography>
                            {inputAnchor.label}
                            {!inputAnchor.optional && <span style={{ color: 'red' }}>&nbsp;*</span>}
                        </Typography>
                    </Box>
                </>
            )}

            {((inputParam && !inputParam.additionalParams) || isAdditionalParams) && (
                <>
                    {inputParam.acceptVariable && (
                        <HandleInputTarget id={inputParam.id} position={position} data={data} isValidConnection={isValidConnection} />
                    )}
                    <ExpandableInput inputParam={inputParam} data={data} />
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

export default NodeInputHandler;

