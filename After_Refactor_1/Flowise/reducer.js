
import { combineReducers } from 'redux';

import customization from './reducers/customization';
import canvas from './reducers/canvas';
import notifier from './reducers/notifier';
import dialog from './reducers/dialog';

const rootReducer = combineReducers({
  customization,
  canvas,
  notifier,
  dialog
});

export default rootReducer;
