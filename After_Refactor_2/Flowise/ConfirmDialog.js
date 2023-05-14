// useConfirm.js
import { useState } from 'react'

const useConfirm = () => {
  const [confirmState, setConfirmState] = useState({
    show: false,
    title: '',
    description: '',
    cancelButtonName: '',
    confirmButtonName: '',
  })

  const showConfirmDialog = (title, description, cancelButtonName, confirmButtonName, onCancel, onConfirm) => {
    setConfirmState({
      show: true,
      title,
      description,
      cancelButtonName,
      confirmButtonName,
      onCancel,
      onConfirm,
    })
  }

  const hideConfirmDialog = () => {
    setConfirmState({
      show: false,
      title: '',
      description: '',
      cancelButtonName: '',
      confirmButtonName: '',
    })
  }

  return { confirmState, showConfirmDialog, hideConfirmDialog }
}

export default useConfirm

// ParentComponent.js
import ConfirmDialog from './ConfirmDialog'

const ParentComponent = () => {
  const { confirmState, showConfirmDialog, hideConfirmDialog } = useConfirm()

  const handleDeleteClick = (id) => {
    showConfirmDialog(
      'Delete Confirmation',
      `Are you sure you want to delete item ${id}?`,
      'Cancel',
      'Delete',
      hideConfirmDialog,
      () => {
        // Delete logic goes here
        hideConfirmDialog()
      }
    )
  }

  return (
    <>
      {list.map((item) => (
        <ListItem key={item.id}>
          <ListItemText primary={item.name} />
          <IconButton aria-label='delete' onClick={() => handleDeleteClick(item.id)}>
            <DeleteIcon />
          </IconButton>
        </ListItem>
      ))}
      <ConfirmDialog confirmState={confirmState} hideConfirmDialog={hideConfirmDialog} />
    </>
  )
}

export default ParentComponent

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