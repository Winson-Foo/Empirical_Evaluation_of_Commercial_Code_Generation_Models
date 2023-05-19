import { useMemo, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { createPortal } from 'react-dom';
import { useSelector } from 'react-redux';
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    Box,
    List,
    ListItemButton,
    ListItem,
    ListItemAvatar,
    ListItemText,
    Typography,
    Stack
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import PerfectScrollbar from 'react-perfect-scrollbar';
import { StyledButton } from 'ui-component/button/StyledButton';
import { DarkCodeEditor } from 'ui-component/editor/DarkCodeEditor';
import { LightCodeEditor } from 'ui-component/editor/LightCodeEditor';
import './EditPromptValuesDialog.css';
import { baseURL } from 'store/constant';

const CodeEditor = ({ disabled, value, onValueChange, placeholder, type, onMouseUp, onBlur, style }) => {
    const customization = useSelector((state) => state.customization);
    const theme = useTheme();

    const Editor = customization.isDarkMode ? DarkCodeEditor : LightCodeEditor;

    return (
        <Editor
            disabled={disabled}
            value={value}
            onValueChange={onValueChange}
            placeholder={placeholder}
            type={type}
            onMouseUp={onMouseUp}
            onBlur={onBlur}
            style={{ fontSize: '0.875rem', minHeight: 'calc(100vh - 220px)', width: '100%', ...style }}
        />
    );
};

CodeEditor.propTypes = {
    disabled: PropTypes.bool,
    value: PropTypes.string,
    onValueChange: PropTypes.func,
    placeholder: PropTypes.string,
    type: PropTypes.string,
    onMouseUp: PropTypes.func,
    onBlur: PropTypes.func,
    style: PropTypes.object
};

const VariableSelector = ({ availableNodes, onSelectOutputResponseClick, disabled }) => (
    <div style={{ flex: 30 }}>
        <Stack flexDirection="row" sx={{ mb: 1, ml: 2 }}>
            <Typography variant="h4">Select Variable</Typography>
        </Stack>
        <PerfectScrollbar style={{ height: '100%', maxHeight: 'calc(100vh - 220px)', overflowX: 'hidden' }}>
            <Box sx={{ pl: 2, pr: 2 }}>
                <List>
                    <ListItemButton
                        sx={{
                            p: 0,
                            borderRadius: `${customization.borderRadius}px`,
                            boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
                            mb: 1
                        }}
                        disabled={disabled}
                        onClick={() => onSelectOutputResponseClick(null, true)}
                    >
                        <ListItem alignItems="center">
                            <ListItemAvatar>
                                <div
                                    style={{
                                        width: 50,
                                        height: 50,
                                        borderRadius: '50%',
                                        backgroundColor: 'white'
                                    }}
                                >
                                    <img
                                        style={{
                                            width: '100%',
                                            height: '100%',
                                            padding: 10,
                                            objectFit: 'contain'
                                        }}
                                        alt="AI"
                                        src="https://raw.githubusercontent.com/zahidkhawaja/langchain-chat-nextjs/main/public/parroticon.png"
                                    />
                                </div>
                            </ListItemAvatar>
                            <ListItemText
                                sx={{ ml: 1 }}
                                primary="question"
                                secondary={`User's question from chatbox`}
                            />
                        </ListItem>
                    </ListItemButton>
                    {availableNodes &&
                        availableNodes.map((node, index) => {
                            const selectedOutputAnchor = node.data.outputAnchors[0].options.find(
                                (ancr) => ancr.name === node.data.outputs['output']
                            );
                            return (
                                <ListItemButton
                                    key={index}
                                    sx={{
                                        p: 0,
                                        borderRadius: `${customization.borderRadius}px`,
                                        boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
                                        mb: 1
                                    }}
                                    disabled={disabled}
                                    onClick={() => onSelectOutputResponseClick(node)}
                                >
                                    <ListItem alignItems="center">
                                        <ListItemAvatar>
                                            <div
                                                style={{
                                                    width: 50,
                                                    height: 50,
                                                    borderRadius: '50%',
                                                    backgroundColor: 'white'
                                                }}
                                            >
                                                <img
                                                    style={{
                                                        width: '100%',
                                                        height: '100%',
                                                        padding: 10,
                                                        objectFit: 'contain'
                                                    }}
                                                    alt={node.data.name}
                                                    src={`${baseURL}/api/v1/node-icon/${node.data.name}`}
                                                />
                                            </div>
                                        </ListItemAvatar>
                                        <ListItemText
                                            sx={{ ml: 1 }}
                                            primary={node.data.inputs.chainName || node.data.id}
                                            secondary={`${selectedOutputAnchor?.label ?? 'output'} from ${node.data.label}`}
                                        />
                                    </ListItem>
                                </ListItemButton>
                            );
                        })}
                </List>
            </Box>
        </PerfectScrollbar>
    </div>
);

VariableSelector.proptypes = {
    availableNodes: PropTypes.array,
    onSelectOutputResponseClick: PropTypes.func,
    disabled: PropTypes.bool
};

const EditPromptValuesDialog = ({ show, dialogProps, onCancel, onConfirm }) => {
    const [inputValue, setInputValue] = useState('');
    const [inputParam, setInputParam] = useState(null);
    const [textCursorPosition, setTextCursorPosition] = useState({});
    const availableNodesForVariable = useMemo(
        () => dialogProps.availableNodesForVariable || [],
        [dialogProps.availableNodesForVariable]
    );

    useEffect(() => {
        if (dialogProps.value) setInputValue(dialogProps.value);
        if (dialogProps.inputParam) setInputParam(dialogProps.inputParam);

        return () => {
            setInputValue('');
            setInputParam(null);
            setTextCursorPosition({});
        };
    }, [dialogProps]);

    const onMouseUp = (e) => {
        if (e.target && e.target.selectionEnd && e.target.value) {
            const cursorPosition = e.target.selectionEnd;
            const textBeforeCursorPosition = e.target.value.substring(0, cursorPosition);
            const textAfterCursorPosition = e.target.value.substring(cursorPosition, e.target.value.length);
            const body = {
                textBeforeCursorPosition,
                textAfterCursorPosition
            };
            setTextCursorPosition(body);
        } else {
            setTextCursorPosition({});
        }
    };

    const onSelectOutputResponseClick = (node, isUserQuestion = false) => {
        const variablePath = isUserQuestion ? 'question' : `${node.id}.data.instance`;

        if (textCursorPosition) {
            const { textBeforeCursorPosition, textAfterCursorPosition } = textCursorPosition;
            const newInput =
                textBeforeCursorPosition === undefined && textAfterCursorPosition === undefined
                    ? `${inputValue}${`{{${variablePath}}}`}`
                    : `${textBeforeCursorPosition}{{${variablePath}}}${textAfterCursorPosition}`;

            setInputValue(newInput);
        }
    };

    const portalElement = document.getElementById('portal');

    const editorProps = {
        disabled: dialogProps.disabled,
        value: inputValue,
        placeholder: inputParam.placeholder,
        type: 'json',
        onMouseUp,
        onBlur: onMouseUp
    };

    const codeEditor = inputParam?.type === 'string' ? (
        <div style={{ flex: 70 }}>
            <Typography sx={{ mb: 2, ml: 1 }} variant="h4">
                {inputParam.label}
            </Typography>
            <PerfectScrollbar
                style={{
                    border: '1px solid',
                    borderColor: theme.palette.grey['500'],
                    borderRadius: '12px',
                    height: '100%',
                    maxHeight: 'calc(100vh - 220px)',
                    overflowX: 'hidden',
                    backgroundColor: 'white'
                }}
            >
                <CodeEditor {...editorProps} />
            </PerfectScrollbar>
        </div>
    ) : null;

    const variableSelector = !dialogProps.disabled && inputParam?.acceptVariable ? (
        <VariableSelector
            availableNodes={availableNodesForVariable}
            onSelectOutputResponseClick={onSelectOutputResponseClick}
            disabled={dialogProps.disabled}
        />
    ) : null;

    const component = show ? (
        <Dialog
            open={show}
            fullWidth
            maxWidth="md"
            aria-labelledby="alert-dialog-title"
            aria-describedby="alert-dialog-description"
        >
            <DialogContent>
                <div style={{ display: 'flex', flexDirection: 'row' }}>
                    {codeEditor}
                    {variableSelector}
                </div>
            </DialogContent>
            <DialogActions>
                <Button onClick={onCancel}>{dialogProps.cancelButtonName}</Button>
                <StyledButton
                    disabled={dialogProps.disabled}
                    variant="contained"
                    onClick={() => onConfirm(inputValue, inputParam.name)}
                >
                    {dialogProps.confirmButtonName}
                </StyledButton>
            </DialogActions>
        </Dialog>
    ) : null;

    return createPortal(component, portalElement);
};

EditPromptValuesDialog.propTypes = {
    show: PropTypes.bool,
    dialogProps: PropTypes.object,
    onCancel: PropTypes.func,
    onConfirm: PropTypes.func
};

export default EditPromptValuesDialog;

