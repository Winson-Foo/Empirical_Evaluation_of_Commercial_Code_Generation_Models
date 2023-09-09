import { createPortal } from 'react-dom'
import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'

import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  OutlinedInput,
  DialogTitle
} from '@mui/material'
import { StyledButton } from 'ui-component/button/StyledButton'

const ChatflowNameInput = ({ value, onChange }) => (
  <OutlinedInput
    sx={{ mt: 1 }}
    id='chatflow-name'
    type='text'
    fullWidth
    placeholder='My New Chatflow'
    value={value}
    onChange={(e) => onChange(e.target.value)}
  />
)

ChatflowNameInput.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func
}

const SaveChatflowDialog = ({ isOpen = false, onSave = () => {}, onClose = () => {} }) => {
  const portalElement = document.getElementById('portal')

  const [chatflowName, setChatflowName] = useState('')
  const [isSaveButtonDisabled, setIsSaveButtonDisabled] = useState(true)

  useEffect(() => {
    setIsSaveButtonDisabled(!chatflowName)
  }, [chatflowName])

  const handleSave = () => {
    onSave(chatflowName)
    setChatflowName('')
    onClose()
  }

  return createPortal(
    <Dialog
      open={isOpen}
      fullWidth
      maxWidth='xs'
      onClose={onClose}
      aria-labelledby='alert-dialog-title'
      aria-describedby='alert-dialog-description'
    >
      <DialogTitle sx={{ fontSize: '1rem' }} id='alert-dialog-title'>
        Save Chatflow
      </DialogTitle>
      <DialogContent>
        <ChatflowNameInput value={chatflowName} onChange={setChatflowName} />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <StyledButton disabled={isSaveButtonDisabled} variant='contained' onClick={handleSave}>
          Save
        </StyledButton>
      </DialogActions>
    </Dialog>,
    portalElement
  )
}

SaveChatflowDialog.propTypes = {
  isOpen: PropTypes.bool,
  onSave: PropTypes.func,
  onClose: PropTypes.func
}

export default SaveChatflowDialog

