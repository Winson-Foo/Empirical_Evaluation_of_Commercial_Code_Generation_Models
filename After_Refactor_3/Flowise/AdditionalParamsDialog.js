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