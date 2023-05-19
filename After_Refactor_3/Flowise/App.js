// app.jsx
import { useSelector } from 'react-redux'
import { ThemeProvider } from '@mui/material/styles'
import { CssBaseline, StyledEngineProvider } from '@mui/material'
import themes from 'themes'
import NavigationScroll from 'layout/NavigationScroll'
import Routes from 'routes'

const App = () => {
    const customization = useSelector((state) => state.customization)

    return (
        <StyledEngineProvider injectFirst>
            <ThemeProvider theme={themes(customization)}>
                <CssBaseline />
                <NavigationScroll>
                    <Routes />
                </NavigationScroll>
            </ThemeProvider>
        </StyledEngineProvider>
    )
}

export default App

// navigation.jsx
import { Box } from '@mui/system'
import Navbar from 'components/Navbar'

const Navigation = () => {
    return (
        <Box sx={{ flexGrow: 1 }}>
            <Navbar />
        </Box>
    )
}

export default Navigation

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

// home.jsx
import { Typography } from '@mui/material'

const Home = () => {
    return (
        <Typography variant="h4" component="h1" align="center">
            Welcome to My App!
        </Typography>
    )
}

export default Home

// about.jsx
import { Typography } from '@mui/material'

const About = () => {
    return (
        <Typography variant="h4" component="h1" align="center">
            About My App
        </Typography>
    )
}

export default About

// routes.jsx
import { Switch, Route } from 'react-router-dom'
import Home from 'pages/Home'
import About from 'pages/About'

const Routes = () => {
    return (
        <Switch>
            <Route exact path="/">
                <Home />
            </Route>
            <Route path="/about">
                <About />
            </Route>
        </Switch>
    )
}

export default Routes