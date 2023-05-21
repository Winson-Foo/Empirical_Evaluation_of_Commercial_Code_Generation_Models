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