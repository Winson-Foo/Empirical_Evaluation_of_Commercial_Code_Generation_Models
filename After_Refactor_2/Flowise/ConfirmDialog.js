// ConfirmDialog.js
import { createPortal } from 'react-dom'
import { Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material'
import { StyledButton } from 'ui-component/button/StyledButton'

const ConfirmDialog = ({ confirmState, hideConfirmDialog }) => {
  const portalElement = document.getElementById('portal')

  const handleConfirm = () => {
    confirmState.onConfirm()
    hideConfirmDialog()
  }

  const handleCancel = () => {
    confirmState.onCancel()
    hideConfirmDialog()
  }

  const component = confirmState.show ? (
    <Dialog
      fullWidth
      maxWidth='xs'
      open={confirmState.show}
      onClose={handleCancel}
      aria-labelledby='alert-dialog-title'
      aria-describedby='alert-dialog-description'
    >
      <DialogTitle sx={{ fontSize: '1rem' }} id='alert-dialog-title'>
        {confirmState.title}
      </DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ color: 'black' }} id='alert-dialog-description'>
          {confirmState.description}
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCancel}>{confirmState.cancelButtonName}</Button>
        <StyledButton variant='contained' onClick={handleConfirm}>
          {confirmState.confirmButtonName}
        </StyledButton>
      </DialogActions>
    </Dialog>
  ) : null

  return createPortal(component, portalElement)
}

export default ConfirmDialog