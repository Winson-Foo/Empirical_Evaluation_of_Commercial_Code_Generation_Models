// actionTypes.js
export const MENU_OPEN = 'MENU_OPEN';
export const SET_MENU = 'SET_MENU';
export const SET_FONT_FAMILY = 'SET_FONT_FAMILY';
export const SET_BORDER_RADIUS = 'SET_BORDER_RADIUS';
export const SET_LAYOUT = 'SET_LAYOUT';
export const SET_DARKMODE = 'SET_DARKMODE';

// initialState.js
import config from 'config';

export const initialState = {
  isOpen: [], // for active default menu
  fontFamily: config.fontFamily,
  borderRadius: config.borderRadius,
  opened: true,
  isHorizontal: localStorage.getItem('isHorizontal') === 'true',
  isDarkMode: localStorage.getItem('isDarkMode') === 'true',
};

// reducers.js
import {
  MENU_OPEN,
  SET_MENU,
  SET_FONT_FAMILY,
  SET_BORDER_RADIUS,
  SET_LAYOUT,
  SET_DARKMODE,
} from './actionTypes';
import { initialState } from './initialState';

const isOpenReducer = (state, action) => {
  switch (action.type) {
    case MENU_OPEN:
      return [action.id];
    default:
      return state;
  }
};

const openedReducer = (state, action) => {
  switch (action.type) {
    case SET_MENU:
      return action.opened;
    default:
      return state;
  }
};

const fontFamilyReducer = (state, action) => {
  switch (action.type) {
    case SET_FONT_FAMILY:
      return action.fontFamily;
    default:
      return state;
  }
};

const borderRadiusReducer = (state, action) => {
  switch (action.type) {
    case SET_BORDER_RADIUS:
      return action.borderRadius;
    default:
      return state;
  }
};

const isHorizontalReducer = (state, action) => {
  switch (action.type) {
    case SET_LAYOUT:
      return action.isHorizontal;
    default:
      return state;
  }
};

const isDarkModeReducer = (state, action) => {
  switch (action.type) {
    case SET_DARKMODE:
      return action.isDarkMode;
    default:
      return state;
  }
};

const customizationReducer = (state = initialState, action) => {
  return {
    isOpen: isOpenReducer(state.isOpen, action),
    opened: openedReducer(state.opened, action),
    fontFamily: fontFamilyReducer(state.fontFamily, action),
    borderRadius: borderRadiusReducer(state.borderRadius, action),
    isHorizontal: isHorizontalReducer(state.isHorizontal, action),
    isDarkMode: isDarkModeReducer(state.isDarkMode, action),
  };
};

export default customizationReducer;

