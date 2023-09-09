// rootReducer.js
import { combineReducers } from 'redux'
import customizationReducer from './customizationReducer'
import canvasReducer from './canvasReducer'
import notifierReducer from './notifierReducer'
import dialogReducer from './dialogReducer'

const rootReducer = combineReducers({
  customization: customizationReducer,
  canvas: canvasReducer,
  notifier: notifierReducer,
  dialog: dialogReducer
})

export default rootReducer