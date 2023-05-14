import { useState } from 'react';
import { useSelector } from 'react-redux';

import {
  Box,
  FormControl,
  Popper,
  TextField,
  Typography,
} from '@mui/material';
import Autocomplete, { autocompleteClasses } from '@mui/material/Autocomplete';
import { styled } from '@mui/material/styles';
import PropTypes from 'prop-types';

const StyledPopper = styled(Popper)({
  boxShadow:
    '0px 8px 10px -5px rgb(0 0 0 / 20%), 0px 16px 24px 2px rgb(0 0 0 / 14%), 0px 6px 30px 5px rgb(0 0 0 / 12%)',
  borderRadius: '10px',
  [`& .${autocompleteClasses.listbox}`]: {
    boxSizing: 'border-box',
    '& ul': {
      padding: 10,
      margin: 10,
    },
  },
});

/**
 * Finds the option object that matches the given name.
 * @param {array} options - available options
 * @param {string} name - name of the selected option
 * @returns {object|null} - matching option object or null
 */
const findOptionByName = (options = [], name) =>
  options.find((option) => option.name === name) || null;

/**
 * Returns a default option object.
 * @returns {object} - default option object
 */
const getDefaultOptionObject = () => null;

const Dropdown = ({
  name,
  value,
  options,
  onSelect,
  disabled = false,
  disableClearable = false,
}) => {
  const customization = useSelector((state) => state.customization);
  const defaultOption = getDefaultOptionObject();
  const [selectedValue, setSelectedValue] = useState(
    value ?? 'choose an option'
  );

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size="small">
      <Autocomplete
        id={name}
        disabled={disabled}
        disableClearable={disableClearable}
        size="small"
        options={options || []}
        value={findOptionByName(options, selectedValue) || defaultOption}
        onChange={(e, selectedOption) => {
          const value = selectedOption ? selectedOption.name : '';
          setSelectedValue(value);
          onSelect(value);
        }}
        PopperComponent={StyledPopper}
        renderInput={(params) => <TextField {...params} value={selectedValue} />}
        renderOption={(props, option) => (
          <Box component="li" {...props}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h5">{option.label}</Typography>
              {option.description && (
                <Typography
                  sx={{
                    color: customization.isDarkMode ? '#9e9e9e' : '',
                  }}
                >
                  {option.description}
                </Typography>
              )}
            </div>
          </Box>
        )}
      />
    </FormControl>
  );
};

Dropdown.propTypes = {
  name: PropTypes.string,
  value: PropTypes.string,
  options: PropTypes.array,
  onSelect: PropTypes.func,
  disabled: PropTypes.bool,
  disableClearable: PropTypes.bool,
};

export default Dropdown;

