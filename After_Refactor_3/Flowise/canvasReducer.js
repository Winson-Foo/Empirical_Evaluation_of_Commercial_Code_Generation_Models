// state.js
export const initialState = {
  isDirty: false,
  chatflow: null,
};

// actionTypes.js
export const SET_DIRTY = 'SET_DIRTY';
export const REMOVE_DIRTY = 'REMOVE_DIRTY';
export const SET_CHATFLOW = 'SET_CHATFLOW';

// actions.js
import { SET_DIRTY, REMOVE_DIRTY, SET_CHATFLOW } from './actionTypes';

export const setDirty = () => ({
  type: SET_DIRTY,
});

export const removeDirty = () => ({
  type: REMOVE_DIRTY,
});

export const setChatflow = (chatflow) => ({
  type: SET_CHATFLOW,
  chatflow,
});

// canvasReducer.js
import { initialState } from './state';
import { SET_DIRTY, REMOVE_DIRTY, SET_CHATFLOW } from './actionTypes';

const canvasReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_DIRTY:
      return { ...state, isDirty: true };
    case REMOVE_DIRTY:
      return { ...state, isDirty: false };
    case SET_CHATFLOW:
      return { ...state, chatflow: action.chatflow };
    default:
      return state;
  }
};

export default canvasReducer;