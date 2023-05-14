import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'
import { FormControl } from '@mui/material'
import ReactJson from 'react-json-view'

const JsonEditor = ({ theme, style, src, name, quotesOnKeys, displayDataTypes, enableClipboard, onEdit, onAdd, onDelete }) => {
  const onClipboardCopy = (e) => {
    const src = e.src
    if (Array.isArray(src) || typeof src === 'object') {
      navigator.clipboard.writeText(JSON.stringify(src, null, '  '))
    } else {
      navigator.clipboard.writeText(src)
    }
  }

  return (
    <ReactJson
      theme={theme}
      style={style}
      src={src}
      name={name}
      quotesOnKeys={quotesOnKeys}
      displayDataTypes={displayDataTypes}
      enableClipboard={(e) => onClipboardCopy(e)}
      onEdit={onEdit}
      onAdd={onAdd}
      onDelete={onDelete}
    />
  )
}

export const JsonEditorInput = ({ value, onChange, disabled = false, isDarkMode = false }) => {
  const [myValue, setMyValue] = useState({})

  useEffect(() => {
    setMyValue(value ? JSON.parse(value) : {})
  }, [value])

  const handleEdit = (edit) => {
    setMyValue(edit.updated_src)
    onChange(JSON.stringify(edit.updated_src))
  }

  const handleDelete = (deleteObj) => {
    setMyValue(deleteObj.updated_src)
    onChange(JSON.stringify(deleteObj.updated_src))
  }

  return (
    <>
      <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
        <JsonEditor
          theme={isDarkMode ? 'ocean' : 'rjv-default'}
          style={{ padding: 10, borderRadius: 10 }}
          src={myValue}
          name={null}
          quotesOnKeys={false}
          displayDataTypes={false}
          enableClipboard={!disabled}
          onEdit={handleEdit}
          onDelete={!disabled && handleDelete}
        />
      </FormControl>
    </>
  )
}

JsonEditorInput.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool,
  isDarkMode: PropTypes.bool
}

