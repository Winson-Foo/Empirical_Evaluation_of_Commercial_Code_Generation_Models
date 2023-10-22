import { combineReducers } from 'redux';

// reducers
import customization from './reducers/customizationReducer';
import canvas from './reducers/canvasReducer';
import notifier from './reducers/notifierReducer';
import dialog from './reducers/dialogReducer';

// constants
const reducers = {
  customization,
  canvas,
  notifier,
  dialog,
};

// combine reducers
const reducer = combineReducers(reducers);

export default reducer;

// Descriptions for each reducer
/*
  customization: stores the theme customization settings
  canvas: stores the current layout of the application
  notifier: stores the system notifications and alerts
  dialog: stores the dialog boxes displayed in the application
*/

