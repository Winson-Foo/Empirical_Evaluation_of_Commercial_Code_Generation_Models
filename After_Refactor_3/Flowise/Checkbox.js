import { useState } from 'react';
import PropTypes from 'prop-types';
import { FormControlLabel, Checkbox } from '@mui/material';

const CheckboxInput = ({ value = false, label, onChange, disabled = false }) => {
  const [isChecked, setIsChecked] = useState(value);

  const handleChange = (event) => {
    const isChecked = event.target.checked;
    setIsChecked(isChecked);
    onChange(isChecked);
  };

  return (
    <FormControlLabel
      control={
        <Checkbox
          checked={isChecked}
          onChange={handleChange}
          disabled={disabled}
          sx={{ mt: 1, width: '100%' }}
          size='small'
        />
      }
      label={label}
    />
  );
};

CheckboxInput.propTypes = {
  value: PropTypes.bool,
  label: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
};

CheckboxInput.defaultProps = {
  value: false,
  disabled: false,
};

export default CheckboxInput;