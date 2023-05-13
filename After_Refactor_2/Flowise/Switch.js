import { useState } from 'react'
import PropTypes from 'prop-types'
import { FormControl, Switch } from '@mui/material'

export const SwitchInput = ({ value, onChange, disabled }) => {
    const [isChecked, setIsChecked] = useState(!!value)

    return (
        <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
            <Switch
                disabled={disabled}
                checked={isChecked}
                onChange={(event) => {
                    setIsChecked(event.target.checked)
                    onChange(event.target.checked)
                }}
            />
        </FormControl>
    )
}

SwitchInput.propTypes = {
    value: PropTypes.string.isRequired,
    onChange: PropTypes.func.isRequired,
    disabled: PropTypes.bool
} 