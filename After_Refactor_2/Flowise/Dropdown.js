import { useState } from 'react'
import { useSelector } from 'react-redux'

import { Popper, FormControl, TextField } from '@mui/material'
import { styled } from '@mui/material/styles'
import PropTypes from 'prop-types'

import AutocompleteComponent from './Autocomplete'

const StyledPopper = styled(Popper)({
  boxShadow:
    '0px 8px 10px -5px rgb(0 0 0 / 20%), 0px 16px 24px 2px rgb(0 0 0 / 14%), 0px 6px 30px 5px rgb(0 0 0 / 12%)',
  borderRadius: '10px',
  '& .MuiAutocomplete-listbox': {
    boxSizing: 'border-box',
    '& ul': {
      padding: 10,
      margin: 10
    }
  }
})

const findMatchingOption = (options, value) => options.find((option) => option.name === value)

const renderOption = (props, option) => (
  <li {...props}>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <Typography variant='h5'>{option.label}</Typography>
      {option.description && <Typography>{option.description}</Typography>}
    </div>
  </li>
)

const Dropdown = ({ name, value, options, onSelect, disabled = false, disableClearable = false }) => {
  const customization = useSelector((state) => state.customization)
  let [internalValue, setInternalValue] = useState(value || '')

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
      <AutocompleteComponent
        id={name}
        value={findMatchingOption(options, internalValue)}
        onChange={(e, selection) => {
          const value = selection ? selection.name : ''
          setInternalValue(value)
          onSelect(value)
        }}
        renderOption={renderOption}
        renderInput={(params) => <TextField {...params} value={internalValue} />}
        PopperComponent={StyledPopper}
        options={options}
        disabled={disabled}
        disableClearable={disableClearable}
      />
    </FormControl>
  )
}

Dropdown.propTypes = {
  name: PropTypes.string,
  value: PropTypes.string,
  options: PropTypes.array,
  onSelect: PropTypes.func,
  disabled: PropTypes.bool,
  disableClearable: PropTypes.bool
}

export default Dropdown

import { Autocomplete, ListboxComponentProps } from '@mui/material'
import PropTypes from 'prop-types'

const AutocompleteComponent = ({ value, renderOption, PopperComponent, ...props }) => {
  const ListboxComponent = (props) => {
    const { children, ...other } = props
    return <ul {...other}>{children}</ul>
  }

  ListboxComponent.propTypes = ListboxComponentProps

  return (
    <Autocomplete
      {...props}
      value={value}
      PopperComponent={PopperComponent}
      renderOption={renderOption}
      ListboxComponent={ListboxComponent}
    />
  )
}

AutocompleteComponent.propTypes = {
  value: PropTypes.object,
  renderOption: PropTypes.func,
  PopperComponent: PropTypes.elementType,
  options: PropTypes.array,
  disabled: PropTypes.bool,
  disableClearable: PropTypes.bool
}

export default AutocompleteComponent