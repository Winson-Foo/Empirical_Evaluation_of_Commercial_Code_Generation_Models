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