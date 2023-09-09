import PropTypes from 'prop-types'
import { FormControlLabel, Checkbox } from '@mui/material'

export const CheckboxInput = ({ value, label = "", onChange = () => {}, disabled = false }) => {
  const handleChange = (event) => {
    onChange(event.target.checked)
  }

  return (
    <FormControlLabel
      sx={{ mt: 1, width: '100%' }}
      size='small'
      control={
        <Checkbox
          disabled={disabled}
          checked={value}
          onChange={handleChange}
        />
      }
      label={label}
    />
  )
}

CheckboxInput.propTypes = {
  value: PropTypes.bool.isRequired,
  label: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool
}

