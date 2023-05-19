// ConfirmContextProvider.js
import { useReducer } from 'react'
import PropTypes from 'prop-types'
import ConfirmContext from './ConfirmContext'
import confirmReducer, { initialState } from '../reducers/confirmReducer'

const ConfirmContextProvider = ({ children }) => {
  const [state, dispatch] = useReducer(confirmReducer, initialState)

  return (
    <ConfirmContext.Provider value={{ state, dispatch }}>
      {children}
    </ConfirmContext.Provider>
  )
}

ConfirmContextProvider.propTypes = {
  children: PropTypes.node.isRequired,
}

export default ConfirmContextProvider

// ConfirmContext.js
import { createContext } from 'react'

const ConfirmContext = createContext()

export default ConfirmContext

// confirmReducer.js
export const initialState = {
  isOpen: false,
  title: '',
  message: '',
  confirmLabel: 'Confirm',
  cancelLabel: 'Cancel',
  onConfirm: () => {},
  onCancel: () => {},
}

const confirmReducer = (state, action) => {
  switch (action.type) {
    case 'OPEN_CONFIRM_DIALOG':
      return {
        ...state,
        isOpen: true,
        title: action.payload.title,
        message: action.payload.message,
        confirmLabel: action.payload.confirmLabel,
        cancelLabel: action.payload.cancelLabel,
        onConfirm: action.payload.onConfirm,
        onCancel: action.payload.onCancel,
      }
    case 'CLOSE_CONFIRM_DIALOG':
      return {
        ...state,
        isOpen: false,
      }
    default:
      return state
  }
}

export default confirmReducer