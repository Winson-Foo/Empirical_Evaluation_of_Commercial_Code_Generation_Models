// App.js

import { useSelector } from 'react-redux'
import { ThemeProvider } from '@mui/material/styles'
import { CssBaseline, StyledEngineProvider } from '@mui/material'
import Navigation from 'components/Navigation'
import Routes from 'components/Routes'
import { defaultTheme } from 'themes'
import PropTypes from 'prop-types'

const App = ({ customization }) => {
  const customTheme = defaultTheme(customization)

  return (
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={customTheme}>
        <CssBaseline />
        <Navigation />
        <Routes />
      </ThemeProvider>
    </StyledEngineProvider>
  )
}

App.propTypes = {
  customization: PropTypes.object.isRequired,
}

export default App

// Navigation.js

import React from 'react'
import NavigationScroll from 'layout/NavigationScroll'

const Navigation = () => {
  return <NavigationScroll />
}

export default Navigation

// Routes.js

import React from 'react'
import { Switch, Route } from 'react-router-dom'
import HomePage from 'views/HomePage'
import AboutPage from 'views/AboutPage'
import ContactPage from 'views/ContactPage'

const Routes = () => {
  return (
    <Switch>
      <Route exact path="/" component={HomePage} />
      <Route exact path="/about" component={AboutPage} />
      <Route exact path="/contact" component={ContactPage} />
    </Switch>
  )
}

export default Routes

// themes.js

const defaultTheme = (customization) => {
  return {
    palette: {
      primary: {
        main: customization.primary,
      },
      secondary: {
        main: customization.secondary,
      },
    },
  }
}

export { defaultTheme }

