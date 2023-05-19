// ConfirmDialog.jsx
import React from 'react';
import PropTypes from 'prop-types';
import { Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Button } from '@mui/material';
import { StyledButton } from 'ui-component/button/StyledButton';

const ConfirmDialog = ({ show, title, description, cancelButtonName, confirmButtonName, onCancel, onConfirm }) => (
  <Dialog
    fullWidth
    maxWidth='xs'
    open={show}
    onClose={onCancel}
    aria-labelledby='alert-dialog-title'
    aria-describedby='alert-dialog-description'
  >
    <DialogTitle sx={{ fontSize: '1rem' }} id='alert-dialog-title'>
      {title}
    </DialogTitle>
    <DialogContent>
      <DialogContentText sx={{ color: 'black' }} id='alert-dialog-description'>
        {description}
      </DialogContentText>
    </DialogContent>
    <DialogActions>
      <Button onClick={onCancel}>{cancelButtonName}</Button>
      <StyledButton variant='contained' onClick={onConfirm}>
        {confirmButtonName}
      </StyledButton>
    </DialogActions>
  </Dialog>
);

ConfirmDialog.propTypes = {
  show: PropTypes.bool.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  cancelButtonName: PropTypes.string.isRequired,
  confirmButtonName: PropTypes.string.isRequired,
  onCancel: PropTypes.func.isRequired,
  onConfirm: PropTypes.func.isRequired,
};

export default ConfirmDialog;

// ConfirmDialogContainer.jsx
import React from 'react';
import { createPortal } from 'react-dom';
import useConfirm from 'hooks/useConfirm';
import ConfirmDialog from './ConfirmDialog';
import { CONFIRM_DIALOG_STRINGS } from './constants';

const ConfirmDialogContainer = () => {
  const { onConfirm, onCancel, confirmState } = useConfirm();
  const portalElement = document.getElementById('portal');

  const component = <ConfirmDialog {...confirmState} onCancel={onCancel} onConfirm={onConfirm} />;

  return createPortal(component, portalElement);
};

export default ConfirmDialogContainer;

// constants.js
export const CONFIRM_DIALOG_STRINGS = {
  title: 'Are you sure?',
  description: 'Do you want to continue with this action?',
  cancelButtonName: 'Cancel',
  confirmButtonName: 'Confirm',
};

