import React from 'react'
import App from './App'
import { store } from 'store'
import { createRoot } from 'react-dom/client'

// style + assets
import 'assets/scss/style.scss'

// third party
import { BrowserRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { SnackbarProvider } from 'notistack'
import ConfirmContextProvider from 'store/context/ConfirmContextProvider'
import { ReactFlowContext } from 'store/context/ReactFlowContext'

const container = document.getElementById('root')
const root = createRoot(container)

const ProviderWrapper = ({ children }) => (
  <Provider store={store}>{children}</Provider>
)

const BrowserRouterWrapper = ({ children }) => (
  <BrowserRouter>{children}</BrowserRouter>
)

const SnackbarProviderWrapper = ({ children }) => (
  <SnackbarProvider>{children}</SnackbarProvider>
)

const ConfirmContextProviderWrapper = ({ children }) => (
  <ConfirmContextProvider>{children}</ConfirmContextProvider>
)

const ReactFlowContextWrapper = ({ children }) => (
  <ReactFlowContext>{children}</ReactFlowContext>
)

const AppWrapper = () => (
  <ProviderWrapper>
    <BrowserRouterWrapper>
      <SnackbarProviderWrapper>
        <ConfirmContextProviderWrapper>
          <ReactFlowContextWrapper>
            <App />
          </ReactFlowContextWrapper>
        </ConfirmContextProviderWrapper>
      </SnackbarProviderWrapper>
    </BrowserRouterWrapper>
  </ProviderWrapper>
)

root.render(
  <React.StrictMode>
    <AppWrapper />
  </React.StrictMode>
)