import { useState } from 'react';
import PropTypes from 'prop-types';
import { FormControlLabel, Checkbox } from '@mui/material';

function CheckboxInput({ value, label, onChange, disabled = false }) {
  const [isChecked, setChecked] = useState(value);

  function handleChange(event) {
    const { checked } = event.target;
    setChecked(checked);
    onChange(checked);
  }

  return (
    <FormControlLabel
      sx={{ mt: 1, width: '100%' }}
      size='small'
      control={
        <Checkbox
          checked={isChecked}
          onChange={handleChange}
          disabled={disabled}
        />
      }
      label={label}
    />
  );
}

CheckboxInput.propTypes = {
  value: PropTypes.bool,
  label: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool,
};

export default CheckboxInput;