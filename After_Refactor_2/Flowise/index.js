import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { SnackbarProvider } from 'notistack';
import ConfirmContextProvider from 'store/context/ConfirmContextProvider';
import { ReactFlowContext } from 'store/context/ReactFlowContext';
import ReactDOM from 'react-dom';
import App from './App';
import { store } from 'store';
import 'assets/scss/style.scss';

function AppProviders({ children }) {
  return (
    <Provider store={store}>
      <BrowserRouter>
        <SnackbarProvider>{children}</SnackbarProvider>
      </BrowserRouter>
    </Provider>
  );
}

function ContextProviders({ children }) {
  return (
    <ConfirmContextProvider>
      <ReactFlowContext>{children}</ReactFlowContext>
    </ConfirmContextProvider>
  );
}

const container = document.getElementById('root');
ReactDOM.render(
  <React.StrictMode>
    <AppProviders>
      <ContextProviders>
        <App />
      </ContextProviders>
    </AppProviders>
  </React.StrictMode>,
  container
);

