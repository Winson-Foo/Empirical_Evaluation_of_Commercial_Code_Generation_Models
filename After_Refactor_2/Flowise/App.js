// imports

import { useSelector } from 'react-redux'
import { ThemeProvider } from '@mui/material/styles'
import { CssBaseline, StyledEngineProvider } from '@mui/material'
import NavigationScroll from 'layout/NavigationScroll'
import RouteConfig from 'routes'
import themes from 'themes'

// App function

function App() {
  const customization = useSelector((state) => state.customization)

  const theme = themes(customization)

  return (
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <NavigationScroll>
          <RouteConfig />
        </NavigationScroll>
      </ThemeProvider>
    </StyledEngineProvider>
  )
}

export default App

// Route config

import React from 'react'
import { Switch, Route } from 'react-router-dom'

// route components

import Home from '../components/Home'
import About from '../components/About'
import Contact from '../components/Contact'

function RouteConfig() {
  return (
    <Switch>
      <Route path="/about">
        <About />
      </Route>
      <Route path="/contact">
        <Contact />
      </Route>
      <Route path="/">
        <Home />
      </Route>
    </Switch>
  )
}

export default RouteConfig