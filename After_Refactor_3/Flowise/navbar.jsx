// navbar.jsx
import { Link as RouterLink } from 'react-router-dom'
import { AppBar, Link, Toolbar, Typography } from '@mui/material'

const Navbar = () => {
    return (
        <AppBar position="static">
            <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    My App
                </Typography>
                <Link component={RouterLink} to="/" color="inherit">
                    Home
                </Link>
                <Link component={RouterLink} to="/about" color="inherit">
                    About
                </Link>
            </Toolbar>
        </AppBar>
    )
}

export default Navbar