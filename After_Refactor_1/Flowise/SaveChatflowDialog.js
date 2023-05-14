import React from 'react';
import { createPortal } from 'react-dom';
import PropTypes from 'prop-types';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  OutlinedInput,
} from '@mui/material';
import { StyledButton } from 'ui-component/button/StyledButton';

const StyledDialog = ({
  open,
  title,
  cancelButtonName,
  confirmButtonName,
  children,
  onCancel,
  onConfirm,
}) => (
  <Dialog
    open={open}
    fullWidth
    maxWidth='xs'
    onClose={onCancel}
    aria-labelledby='alert-dialog-title'
    aria-describedby='alert-dialog-description'>
    <DialogTitle sx={{ fontSize: '1rem' }} id='alert-dialog-title'>
      {title}
    </DialogTitle>
    <DialogContent>{children}</DialogContent>
    <DialogActions>
      <Button onClick={onCancel}>{cancelButtonName}</Button>
      <StyledButton variant='contained' onClick={onConfirm}>
        {confirmButtonName}
      </StyledButton>
    </DialogActions>
  </Dialog>
);

StyledDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  title: PropTypes.string.isRequired,
  cancelButtonName: PropTypes.string.isRequired,
  confirmButtonName: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
  onCancel: PropTypes.func.isRequired,
  onConfirm: PropTypes.func.isRequired,
};

const StyledOutlinedInput = ({
  id,
  type,
  fullWidth,
  placeholder,
  value,
  onChange,
}) => (
  <OutlinedInput
    sx={{ mt: 1 }}
    id={id}
    type={type}
    fullWidth={fullWidth}
    placeholder={placeholder}
    value={value}
    onChange={onChange}
  />
);

StyledOutlinedInput.propTypes = {
  id: PropTypes.string.isRequired,
  type: PropTypes.string.isRequired,
  fullWidth: PropTypes.bool.isRequired,
  placeholder: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

const SaveChatflowDialog = (props) => {
  const portalElement = document.getElementById('portal');

  const { show, dialogProps, onCancel, onConfirm } = props;
  const { title, cancelButtonName, confirmButtonName } = dialogProps;

  const [chatflowName, setChatflowName] = React.useState('');

  React.useEffect(() => {
    setIsReadyToSave(chatflowName.length > 0);
  }, [chatflowName]);

  const [isReadyToSave, setIsReadyToSave] = React.useState(false);

  return createPortal(
    <StyledDialog
      open={show}
      title={title}
      cancelButtonName={cancelButtonName}
      confirmButtonName={confirmButtonName}
      onCancel={onCancel}
      onConfirm={() => onConfirm(chatflowName)}>
      <StyledOutlinedInput
        id='chatflow-name'
        type='text'
        fullWidth
        placeholder='My New Chatflow'
        value={chatflowName}
        onChange={(e) => setChatflowName(e.target.value)}
      />
    </StyledDialog>,
    portalElement
  );
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

