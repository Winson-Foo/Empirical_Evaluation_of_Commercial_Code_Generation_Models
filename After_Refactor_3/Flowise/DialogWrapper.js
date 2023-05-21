import PropTypes from 'prop-types'
import { Dialog } from '@mui/material'
import DialogContentWrapper from './DialogContentWrapper'

const DialogWrapper = ({ show, onCancel, dialogProps }) => {
  return (
    <Dialog
      onClose={onCancel}
      open={show}
      fullWidth
      maxWidth='sm'
      aria-labelledby='alert-dialog-title'
      aria-describedby='alert-dialog-description'
    >
      <DialogContentWrapper
        inputParams={dialogProps.inputParams}
        data={dialogProps.data}
        disabled={dialogProps.disabled}
      />
    </Dialog>
  )
}

DialogWrapper.propTypes = {
  show: PropTypes.bool,
  onCancel: PropTypes.func,
  dialogProps: PropTypes.object,
}

export default DialogWrapper