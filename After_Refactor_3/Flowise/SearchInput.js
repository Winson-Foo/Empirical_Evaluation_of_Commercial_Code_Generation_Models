// SearchInput.js

import React from 'react';
import InputAdornment from '@mui/material/InputAdornment';
import IconSearch from '@tabler/icons';

import OutlinedInput from '@mui/material/OutlinedInput';

const SearchInput = ({ value, onChange }) => {
  const theme = useTheme();
  return (
    <OutlinedInput
      sx={{ width: '100%', pr: 1, pl: 2, my: 2 }}
      id='input-search-node'
      value={value}
      onChange={onChange}
      placeholder='Search nodes'
      startAdornment={
        <InputAdornment position='start'>
          <IconSearch
            stroke={1.5}
            size='1rem'
            color={theme.palette.grey[500]}
          />
        </InputAdornment>
      }
      aria-describedby='search-helper-text'
      inputProps={{
        'aria-label': 'weight',
      }}
    />
  );
};

SearchInput.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

export default SearchInput;