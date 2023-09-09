import PropTypes from 'prop-types';
import { Handle, Position, useUpdateNodeInternals } from 'reactflow';
import { useEffect, useRef, useState, useContext } from 'react';

// material-ui
import { useTheme, styled } from '@mui/material/styles';
import { Box, Typography, Tooltip } from '@mui/material';
import { tooltipClasses } from '@mui/material/Tooltip';
import { flowContext } from 'store/context/ReactFlowContext';
import { isValidConnection } from 'utils/genericHelper';
import { Dropdown } from 'ui-component/dropdown/Dropdown';

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 500
    }
});

const NodeOutputHandler = ({ outputAnchor, data, disabled = false }) => {
    const { reactFlowInstance } = useContext(flowContext);
    const updateNodeInternals = useUpdateNodeInternals();
    const theme = useTheme();
    const ref = useRef(null);
    const [position, setPosition] = useState(0);
    const [dropdownValue, setDropdownValue] = useState(null);

    useEffect(() => {
        if (!ref.current || !ref.current.offsetTop || !ref.current.clientHeight) return;

        setTimeout(() => {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2);
            updateNodeInternals(data.id);
        }, 0);
    }, [data.id, ref, updateNodeInternals]);

    useEffect(() => {
        setTimeout(() => {
            updateNodeInternals(data.id);
        }, 0);
    }, [data.id, position, updateNodeInternals]);

    useEffect(() => {
        if (!dropdownValue) return;

        setTimeout(() => {
            updateNodeInternals(data.id);
        }, 0);
    }, [data.id, dropdownValue, updateNodeInternals]);

    const outputOptionsExist = outputAnchor.options && outputAnchor.options.length > 0;
    const selectedOption = outputOptionsExist && outputAnchor.options.find((opt) => opt.name === data.outputs?.[outputAnchor.name]);

    const handleDropdownSelect = (newValue) => {
        setDropdownValue(newValue);
        data.outputs[outputAnchor.name] = newValue;
    }

    const handleGetOutputId = () => {
        return selectedOption?.id ?? '';
    };

    const handleGetTooltipTitle = () => {
        return selectedOption?.type ?? outputAnchor.type;
    }

    if (outputAnchor.type === 'options' && outputOptionsExist) {
        return (
            <div ref={ref}>
                <CustomWidthTooltip placement='right' title={handleGetTooltipTitle()}>
                    <Handle
                        type='source'
                        position={Position.Right}
                        id={handleGetOutputId()}
                        isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                        style={{
                            height: 10,
                            width: 10,
                            backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                            top: position
                        }}
                    />
                </CustomWidthTooltip>
                <Box sx={{ p: 2, textAlign: 'end' }}>
                    <Typography>{outputAnchor.label}</Typography>
                    <Dropdown
                        disabled={disabled}
                        disableClearable={true}
                        name={outputAnchor.name}
                        options={outputAnchor.options}
                        onSelect={handleDropdownSelect}
                        value={data.outputs[outputAnchor.name] ?? outputAnchor.default ?? 'choose an option'}
                    />
                </Box>
            </div>
        );
    }

    if (outputAnchor.type !== 'options' && !outputAnchor.options) {
        return (
            <div ref={ref}>
                <CustomWidthTooltip placement='right' title={outputAnchor.type}>
                    <Handle
                        type='source'
                        position={Position.Right}
                        id={outputAnchor.id}
                        isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                        style={{
                            height: 10,
                            width: 10,
                            backgroundColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                            top: position
                        }}
                    />
                </CustomWidthTooltip>
                <Box sx={{ p: 2, textAlign: 'end' }}>
                    <Typography>{outputAnchor.label}</Typography>
                </Box>
            </div>
        );
    }

    return null;
}

NodeOutputHandler.propTypes = {
    outputAnchor: PropTypes.object,
    data: PropTypes.object,
    disabled: PropTypes.bool
};

export default NodeOutputHandler;

