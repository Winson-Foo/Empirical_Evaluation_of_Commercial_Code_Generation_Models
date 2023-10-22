import { createPortal } from 'react-dom';
import { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import Dialog from './Dialog';
import OutlinedInput from './OutlinedInput';
import { Button, DialogActions, DialogContent, DialogTitle } from '@mui/material';
import { StyledButton } from 'ui-component/button/StyledButton';

const SaveChatflowDialog = ({ show, dialogProps, onCancel, onConfirm }) => {
  const portalElementRef = useRef(document.getElementById('portal'));

  const [chatflowName, setChatflowName] = useState('');
  const [isReadyToSave, setIsReadyToSave] = useState(false);

  useEffect(() => {
    setIsReadyToSave(Boolean(chatflowName));
  }, [chatflowName]);

  const { title, cancelButtonName, confirmButtonName } = dialogProps;

  const component = show ? (
    <Dialog
      open={show}
      fullWidth
      maxWidth="xs"
      onClose={onCancel}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle sx={{ fontSize: '1rem' }} id="alert-dialog-title">
        {title}
      </DialogTitle>
      <DialogContent>
        <OutlinedInput
          sx={{ mt: 1 }}
          id="chatflow-name"
          type="text"
          fullWidth
          placeholder="My New Chatflow"
          value={chatflowName}
          onChange={(e) => setChatflowName(e.target.value)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel}>{cancelButtonName}</Button>
        <StyledButton
          disabled={!isReadyToSave}
          variant="contained"
          onClick={() => onConfirm(chatflowName)}
        >
          {confirmButtonName}
        </StyledButton>
      </DialogActions>
    </Dialog>
  ) : null;

  return createPortal(component, portalElementRef.current);
};

SaveChatflowDialog.propTypes = {
  show: PropTypes.bool.isRequired,
  dialogProps: PropTypes.shape({
    title: PropTypes.string.isRequired,
    cancelButtonName: PropTypes.string.isRequired,
    confirmButtonName: PropTypes.string.isRequired,
  }).isRequired,
  onCancel: PropTypes.func.isRequired,
  onConfirm: PropTypes.func.isRequired,
};

export default SaveChatflowDialog;

