import { useState } from 'react'
import PropTypes from 'prop-types'
import { FormControl, Switch } from '@mui/material'

const SwitchInput = ({ value = false, onChange, disabled = false }) => {
  const [myValue, setMyValue] = useState(!!value)

  const handleChange = (event) => {
    const newValue = event.target.checked
    setMyValue(newValue)
    onChange(newValue)
  }

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
      <Switch
        disabled={disabled}
        checked={myValue}
        onChange={handleChange}
      />
    </FormControl>
  )
}

SwitchInput.propTypes = {
  value: PropTypes.bool,
  onChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool
}

export default SwitchInput

