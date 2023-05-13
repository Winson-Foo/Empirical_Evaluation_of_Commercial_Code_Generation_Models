// customizationReducer.js
const customizationReducer = (state = {}, action) => {
  // reducer logic here
}

export default customizationReducer

// canvasReducer.js
const canvasReducer = (state = {}, action) => {
  // reducer logic here
}

export default canvasReducer

// notifierReducer.js
const notifierReducer = (state = {}, action) => {
  // reducer logic here
}

export default notifierReducer

// dialogReducer.js
const dialogReducer = (state = {}, action) => {
  // reducer logic here
}

export default dialogReducer

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