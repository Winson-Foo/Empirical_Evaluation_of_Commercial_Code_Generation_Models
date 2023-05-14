import { useReducer } from 'react'
import PropTypes from 'prop-types'
import ConfirmContext from './ConfirmContext'
import alertReducer, { initialState } from '../reducers/dialogReducer'

/**
 * The ConfirmContextProvider component is responsible for providing the ConfirmContext to its child components.
 * @param {any} children - The child components of this provider.
 * @returns {JSX.Element} - The ConfirmContext.Provider component.
 */
const ConfirmContextProvider = ({ children }) => {
  // We use useReducer hook to manage the state of the ConfirmContext
  const [dialogState, dialogDispatch] = useReducer(alertReducer, initialState)

  return (
    <ConfirmContext.Provider value={[dialogState, dialogDispatch]}>
      {children}
    </ConfirmContext.Provider>
  )
}

// We use PropTypes to specify the expected input props.
ConfirmContextProvider.propTypes = {
  children: PropTypes.node.isRequired,
}

export default ConfirmContextProvider

