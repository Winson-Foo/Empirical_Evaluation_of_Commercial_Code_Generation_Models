import { FormControl, OutlinedInput } from '@mui/material'
import PropTypes from 'prop-types'
import EditPromptValuesDialog from 'ui-component/dialog/EditPromptValuesDialog'

const Input = ({ 
  inputParam,
  value,
  onChange,
  disabled = false,
  showDialog,
  dialogProps,
  onDialogCancel,
  onDialogConfirm 
}) => {

  const getInputType = () => {
    switch (inputParam.type) {
      case 'string':
        return 'text'
      case 'password':
        return 'password'
      case 'number':
        return 'number'
      default:
        return 'text'
    }
  }

  const getRows = () => {
    return inputParam.rows ?? 1
  }

  const handleInputChange = (e) => {
    onChange(e.target.value)
  }

  return (
    <>
      <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
        <OutlinedInput
          id={inputParam.name}
          size='small'
          disabled={disabled}
          type={getInputType()}
          placeholder={inputParam.placeholder}
          multiline={!!inputParam.rows}
          rows={getRows()}
          value={value}
          name={inputParam.name}
          onChange={handleInputChange}
          inputProps={{
            style: {
              height: getRows() ? '90px' : 'inherit'
            }
          }}
        />
      </FormControl>
      <EditPromptValuesDialog
        show={showDialog}
        dialogProps={dialogProps}
        onCancel={onDialogCancel}
        onConfirm={(newValue, inputParamName) => {
          onChange(newValue)
          onDialogConfirm(newValue, inputParamName)
        }}
      ></EditPromptValuesDialog>
    </>
  )
}

Input.propTypes = {
  inputParam: PropTypes.object,
  value: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool,
  showDialog: PropTypes.bool,
  dialogProps: PropTypes.object,
  onDialogCancel: PropTypes.func,
  onDialogConfirm: PropTypes.func
}

export default Input

import React from 'react'
import PropTypes from 'prop-types'
import { Dialog, DialogTitle } from '@mui/material'

const EditPromptValuesDialog = ({
  show,
  dialogProps,
  onCancel,
  onConfirm
}) => {
  const handleCancel = () => {
    onCancel()
  }

  const handleConfirm = () => {
    onConfirm(dialogProps.value, dialogProps.inputParamName)
  }

  return (
    <Dialog open={show} onClose={handleCancel}>
      <DialogTitle>{dialogProps.title}</DialogTitle>
      {dialogProps.content}
      <Button onClick={handleCancel}>Cancel</Button>
      <Button onClick={handleConfirm}>Confirm</Button>
    </Dialog>
  )
}

EditPromptValuesDialog.propTypes = {
  show: PropTypes.bool,
  dialogProps: PropTypes.object,
  onCancel: PropTypes.func,
  onConfirm: PropTypes.func
}

export default EditPromptValuesDialog