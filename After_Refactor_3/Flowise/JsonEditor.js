import { useState } from 'react'
import PropTypes from 'prop-types'
import { FormControl } from '@mui/material'
import ReactJson from 'react-json-view'

const onClipboardCopy = (e) => {
  const src = e.src
  if (Array.isArray(src) || typeof src === 'object') {
    navigator.clipboard.writeText(JSON.stringify(src, null, '  '))
  } else {
    navigator.clipboard.writeText(src)
  }
}

const JsonEditorInput = ({ value = {}, onChange, disabled = false, isDarkMode = false }) => {
  const [myValue, setMyValue] = useState(value)

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
      <ReactJson
        theme={isDarkMode ? 'ocean' : 'rjv-default'}
        style={{ padding: 10, borderRadius: 10 }}
        src={myValue}
        name={null}
        quotesOnKeys={false}
        displayDataTypes={false}
        enableClipboard={(e) => onClipboardCopy(e)}
        onEdit={(edit) => {
          setMyValue(edit.updated_src)
          onChange(JSON.stringify(edit.updated_src))
        }}
        onAdd={() => {
          //console.log(add)
        }}
        onDelete={(deleteobj) => {
          setMyValue(deleteobj.updated_src)
          onChange(JSON.stringify(deleteobj.updated_src))
        }}
        {...(!disabled ? {} : { enableClipboard: false })}
      />
    </FormControl>
  )
}

JsonEditorInput.propTypes = {
  value: PropTypes.object,
  onChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  isDarkMode: PropTypes.bool
}

export default JsonEditorInput

