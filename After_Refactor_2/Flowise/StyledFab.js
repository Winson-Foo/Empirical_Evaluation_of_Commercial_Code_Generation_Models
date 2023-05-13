// Button.js
import { styled } from '@mui/material/styles'
import { Fab } from '@mui/material'

const Button = styled(Fab)(({ theme, color = 'primary' }) => ({
    color: 'white',
    backgroundColor: theme.palette[color].main,
    '&:hover': {
        backgroundColor: theme.palette[color].main,
        backgroundImage: `linear-gradient(rgb(0 0 0/10%) 0 0)`
    }
}))

export default Button

// Usage
import Button from './Button'

<Button color="secondary">Click me</Button>