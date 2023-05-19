import PropTypes from 'prop-types'
import NodeInputHandler from 'views/canvas/NodeInputHandler'

const InputField = ({ disabled, inputParam, data }) => {
  return (
    <NodeInputHandler
      disabled={disabled}
      inputParam={inputParam}
      data={data}
      isAdditionalParams={true}
    />
  )
}

InputField.propTypes = {
  disabled: PropTypes.bool,
  inputParam: PropTypes.object,
  data: PropTypes.object,
}

export default InputField

import PerfectScrollbar from 'react-perfect-scrollbar'
import PropTypes from 'prop-types'
import InputField from './InputField'

const DialogContentWrapper = ({ inputParams, data, disabled }) => {
  return (
    <PerfectScrollbar
      style={{
        height: '100%',
        maxHeight: 'calc(100vh - 220px)',
        overflowX: 'hidden',
      }}
    >
      {inputParams.map((inputParam, index) => (
        <InputField
          disabled={disabled}
          key={index}
          inputParam={inputParam}
          data={data}
        />
      ))}
    </PerfectScrollbar>
  )
}

DialogContentWrapper.propTypes = {
  inputParams: PropTypes.array,
  data: PropTypes.object,
  disabled: PropTypes.bool,
}

export default DialogContentWrapper

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

import PropTypes from 'prop-types'
import { createPortal } from 'react-dom'
import { useState, useEffect } from 'react'
import DialogWrapper from './DialogWrapper'

const AdditionalParamsDialog = ({ show, dialogProps, onCancel }) => {
  const portalElement = document.getElementById('portal')
  const [inputParams, setInputParams] = useState([])
  const [data, setData] = useState({})

  useEffect(() => {
    if (dialogProps.inputParams) setInputParams(dialogProps.inputParams)
    if (dialogProps.data) setData(dialogProps.data)
  }, [dialogProps])

  const component = createPortal(
    <DialogWrapper show={show} onCancel={onCancel} dialogProps={dialogProps} />,
    portalElement
  )

  return component
}

AdditionalParamsDialog.propTypes = {
  show: PropTypes.bool,
  dialogProps: PropTypes.object,
  onCancel: PropTypes.func,
}

export default AdditionalParamsDialog