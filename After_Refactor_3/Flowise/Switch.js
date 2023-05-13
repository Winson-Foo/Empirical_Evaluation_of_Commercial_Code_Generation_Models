// This component renders a Switch input field
export const SwitchInput = ({ value, onChange, disabled = false }) => {
    
    // Initializes state for the Switch input field
    const [myValue, setMyValue] = useState(!!value)
    
    // Renders the Switch input field
    return (
        <>
            <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
                <Switch
                    // Disables the Switch input field if 'disabled' prop is true
                    disabled={disabled}
                    // Sets the current value of the Switch input field to 'myValue'
                    checked={myValue}
                    // Sets the new value of the Switch input field to 'myValue' and calls the 'onChange' prop function with the new value
                    onChange={({ target: { checked } }) => {
                        setMyValue(checked)
                        onChange(checked)
                    }}
                />
            </FormControl>
        </>
    )
}

// Specifies the type of props passed to the component
SwitchInput.propTypes = {
    value: PropTypes.string,
    onChange: PropTypes.func,
    disabled: PropTypes.bool
}