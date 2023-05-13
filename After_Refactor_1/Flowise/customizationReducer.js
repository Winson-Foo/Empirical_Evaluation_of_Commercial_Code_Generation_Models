// project imports
import config from 'config'

// action - state management
import * as actionTypes from '../actions'

const localStorageIsHorizontal = localStorage.getItem('isHorizontal') === 'true'
const localStorageIsDarkMode = localStorage.getItem('isDarkMode') === 'true'

const initialState = {
  isOpen: [], // for active default menu
  fontFamily: config.fontFamily,
  borderRadius: config.borderRadius,
  opened: true,
  isHorizontal: localStorageIsHorizontal,
  isDarkMode: localStorageIsDarkMode
}

const customizationReducer = (state = initialState, action) => {
  const { type, id, opened, fontFamily, borderRadius, isHorizontal, isDarkMode } = action
  switch (type) {
    case actionTypes.MENU_OPEN:
      return {
        ...state,
        isOpen: [id]
      }
    case actionTypes.SET_MENU:
      return {
        ...state,
        opened
      }
    case actionTypes.SET_FONT_FAMILY:
      return {
        ...state,
        fontFamily
      }
    case actionTypes.SET_BORDER_RADIUS:
      return {
        ...state,
        borderRadius
      }
    case actionTypes.SET_LAYOUT:
      return {
        ...state,
        isHorizontal
      }
    case actionTypes.SET_DARKMODE:
      return {
        ...state,
        isDarkMode
      }
    default:
      return state
  }
}

export default customizationReducer

