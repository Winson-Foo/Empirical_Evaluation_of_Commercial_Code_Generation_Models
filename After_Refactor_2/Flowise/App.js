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