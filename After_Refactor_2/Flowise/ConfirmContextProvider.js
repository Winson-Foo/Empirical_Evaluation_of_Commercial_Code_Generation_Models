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