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