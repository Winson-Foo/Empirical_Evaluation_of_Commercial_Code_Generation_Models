import { createPortal } from 'react-dom';
import { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import PropTypes from 'prop-types';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItem from '@mui/material/ListItem';
import ListItemAvatar from '@mui/material/ListItemAvatar';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import PerfectScrollbar from 'react-perfect-scrollbar';
import { StyledButton } from 'ui-component/button/StyledButton';
import { DarkCodeEditor } from 'ui-component/editor/DarkCodeEditor';
import { LightCodeEditor } from 'ui-component/editor/LightCodeEditor';
import './EditPromptValuesDialog.css';
import { baseURL } from 'store/constant';

const PortalElement = document.getElementById('portal');

const DarkCodeEditorWrapper = (props) => {
  const customization = useSelector((state) => state.customization);
  return (
    <DarkCodeEditor
      {...props}
      style={{
        fontSize: '0.875rem',
        minHeight: 'calc(100vh - 220px)',
        width: '100%',
      }}
      theme={customization.isDarkMode ? 'dark' : 'light'}
    />
  );
};

const LightCodeEditorWrapper = (props) => {
  const customization = useSelector((state) => state.customization);
  return (
    <LightCodeEditor
      {...props}
      style={{
        fontSize: '0.875rem',
        minHeight: 'calc(100vh - 220px)',
        width: '100%',
      }}
      theme={customization.isDarkMode ? 'dark' : 'light'}
    />
  );
};

const AvailableNodesList = ({ nodes, inputParam, onSelectOutputResponseClick }) => {
  const customization = useSelector((state) => state.customization);
  const theme = useTheme();

  return (
    <PerfectScrollbar style={{ height: '100%', maxHeight: 'calc(100vh - 220px)', overflowX: 'hidden' }}>
      <Box sx={{ pl: 2, pr: 2 }}>
        <List>
          <ListItemButton
            sx={{
              p: 0,
              borderRadius: `${customization.borderRadius}px`,
              boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
              mb: 1,
            }}
            disabled={inputParam.disabled}
            onClick={() => onSelectOutputResponseClick(null, true)}
          >
            <ListItem alignItems='center'>
              <ListItemAvatar>
                <div
                  style={{
                    width: 50,
                    height: 50,
                    borderRadius: '50%',
                    backgroundColor: 'white',
                  }}
                >
                  <img
                    style={{
                      width: '100%',
                      height: '100%',
                      padding: 10,
                      objectFit: 'contain',
                    }}
                    alt='AI'
                    src='https://raw.githubusercontent.com/zahidkhawaja/langchain-chat-nextjs/main/public/parroticon.png'
                  />
                </div>
              </ListItemAvatar>
              <ListItemText sx={{ ml: 1 }} primary='question' secondary={`User's question from chatbox`} />
            </ListItem>
          </ListItemButton>
          {nodes.map((node, index) => {
            const selectedOutputAnchor = node.data.outputAnchors[0].options.find((ancr) => ancr.name === node.data.outputs['output']);

            return (
              <ListItemButton
                key={index}
                sx={{
                  p: 0,
                  borderRadius: `${customization.borderRadius}px`,
                  boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
                  mb: 1,
                }}
                disabled={inputParam.disabled}
                onClick={() => onSelectOutputResponseClick(node)}
              >
                <ListItem alignItems='center'>
                  <ListItemAvatar>
                    <div
                      style={{
                        width: 50,
                        height: 50,
                        borderRadius: '50%',
                        backgroundColor: 'white',
                      }}
                    >
                      <img
                        style={{
                          width: '100%',
                          height: '100%',
                          padding: 10,
                          objectFit: 'contain',
                        }}
                        alt={node.data.name}
                        src={`${baseURL}/api/v1/node-icon/${node.data.name}`}
                      />
                    </div>
                  </ListItemAvatar>
                  <ListItemText
                    sx={{ ml: 1 }}
                    primary={node.data.inputs.chainName ? node.data.inputs.chainName : node.data.id}
                    secondary={`${selectedOutputAnchor?.label ?? 'output'} from ${node.data.label}`}
                  />
                </ListItem>
              </ListItemButton>
            );
          })}
        </List>
      </Box>
    </PerfectScrollbar>
  );
};

const EditPromptValuesDialog = ({ show, dialogProps, onCancel, onConfirm }) => {
  const theme = useTheme();

  const [inputValue, setInputValue] = useState('');
  const [inputParam, setInputParam] = useState(null);
  const [textCursorPosition, setTextCursorPosition] = useState({});

  useEffect(() => {
    if (dialogProps.value || dialogProps.inputParam) {
      setInputValue(dialogProps.value);
      setInputParam(dialogProps.inputParam);
    }

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
        textAfterCursorPosition,
      };
      setTextCursorPosition(body);
    } else {
      setTextCursorPosition({});
    }
  };

  const onSelectOutputResponseClick = (node, isUserQuestion = false) => {
    if (!textCursorPosition) return;

    let variablePath = isUserQuestion ? `question` : `${node.id}.data.instance`;
    let newInput = inputValue;

    if (textCursorPosition.textBeforeCursorPosition === undefined && textCursorPosition.textAfterCursorPosition === undefined) {
      newInput = `${inputValue}${`{{${variablePath}}}`}`;
    } else {
      newInput = `${textCursorPosition.textBeforeCursorPosition}{{${variablePath}}}${textCursorPosition.textAfterCursorPosition}`;
    }

    setInputValue(newInput);
  };

  const renderEditor = () => (
    <div style={{ flex: 70 }}>
      <Typography sx={{ mb: 2, ml: 1 }} variant='h4'>
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
          backgroundColor: 'white',
        }}
      >
        {inputParam.type === 'string' ? (
          <DarkCodeEditorWrapper
            disabled={dialogProps.disabled}
            value={inputValue}
            onValueChange={(code) => setInputValue(code)}
            placeholder={inputParam.placeholder}
            type='json'
            onMouseUp={(e) => onMouseUp(e)}
            onBlur={(e) => onMouseUp(e)}
          />
        ) : (
          <LightCodeEditorWrapper
            disabled={dialogProps.disabled}
            value={inputValue}
            onValueChange={(code) => setInputValue(code)}
            placeholder={inputParam.placeholder}
            type='json'
            onMouseUp={(e) => onMouseUp(e)}
            onBlur={(e) => onMouseUp(e)}
          />
        )}
      </PerfectScrollbar>
    </div>
  );

  const renderVariablesList = () => (
    <div style={{ flex: 30 }}>
      <Stack flexDirection='row' sx={{ mb: 1, ml: 2 }}>
        <Typography variant='h4'>Select Variable</Typography>
      </Stack>
      <AvailableNodesList
        nodes={dialogProps.availableNodesForVariable}
        inputParam={dialogProps}
        onSelectOutputResponseClick={onSelectOutputResponseClick}
      />
    </div>
  );

  const component = show && (
    <Dialog open={show} fullWidth maxWidth='md' aria-labelledby='alert-dialog-title' aria-describedby='alert-dialog-description'>
      <DialogContent>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
          {inputParam && renderEditor()}
          {inputParam?.acceptVariable && renderVariablesList()}
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel}>{dialogProps.cancelButtonName}</Button>
        <StyledButton disabled={dialogProps.disabled} variant='contained' onClick={() => onConfirm(inputValue, inputParam.name)}>
          {dialogProps.confirmButtonName}
        </StyledButton>
      </DialogActions>
    </Dialog>
  );

  return createPortal(component, PortalElement);
};

EditPromptValuesDialog.propTypes = {
  show: PropTypes.bool,
  dialogProps: PropTypes.shape({
    disabled: PropTypes.bool.isRequired,
    value: PropTypes.string.isRequired,
    inputParam: PropTypes.shape({
      label: PropTypes.string.isRequired,
      placeholder: PropTypes.string.isRequired,
      type: PropTypes.oneOf(['string']).isRequired,
      name: PropTypes.string.isRequired,
      acceptVariable: PropTypes.bool.isRequired,
      disabled: PropTypes.bool.isRequired,
    }),
    availableNodesForVariable: PropTypes.arrayOf(PropTypes.object),
    cancelButtonName: PropTypes.string.isRequired,
    confirmButtonName: PropTypes.string.isRequired,
  }),
  onCancel: PropTypes.func.isRequired,
  onConfirm: PropTypes.func.isRequired,
};

export default EditPromptValuesDialog;