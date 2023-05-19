// actionTypes.js

export const SET_DIRTY = 'SET_DIRTY';
export const REMOVE_DIRTY = 'REMOVE_DIRTY';
export const SET_CHATFLOW = 'SET_CHATFLOW';

import * as actionTypes from '../actionTypes';

const canvasReducer = (state = initialState, action) => {
    const { type, chatflow } = action;
    switch (type) {
        case actionTypes.SET_DIRTY:
            return { ...state, isDirty: true };
        case actionTypes.REMOVE_DIRTY:
            return { ...state, isDirty: false };
        case actionTypes.SET_CHATFLOW:
            return { ...state, chatflow };
        default:
            return state;
    }
};

// actions.js

import * as actionTypes from './actionTypes';

export const setDirty = () => ({ type: actionTypes.SET_DIRTY });
export const removeDirty = () => ({ type: actionTypes.REMOVE_DIRTY });
export const setChatFlow = chatflow => ({ type: actionTypes.SET_CHATFLOW, chatflow });

import { connect } from 'react-redux';
import { setDirty, removeDirty, setChatFlow } from '../actions';

const MyComponent = ({ isDirty, chatflow, setDirty, removeDirty, setChatFlow }) => {
  // component logic here
}

const mapStateToProps = state => ({
  isDirty: state.canvasReducer.isDirty,
  chatflow: state.canvasReducer.chatflow,
});

const mapDispatchToProps = {
  setDirty,
  removeDirty,
  setChatFlow,
};

export default connect(mapStateToProps, mapDispatchToProps)(MyComponent);

// actionTypes.js

export const SET_DIRTY = 'SET_DIRTY';
export const REMOVE_DIRTY = 'REMOVE_DIRTY';
export const SET_CHATFLOW = 'SET_CHATFLOW';

// actions.js

import * as actionTypes from './actionTypes';

export const setDirty = () => ({ type: actionTypes.SET_DIRTY });
export const removeDirty = () => ({ type: actionTypes.REMOVE_DIRTY });
export const setChatFlow = chatflow => ({ type: actionTypes.SET_CHATFLOW, chatflow });

// canvasReducer.js

import * as actionTypes from '../actionTypes';

const initialState = {
    isDirty: false,
    chatflow: null
}

const canvasReducer = (state = initialState, action) => {
    const { type, chatflow } = action;
    switch (type) {
        case actionTypes.SET_DIRTY:
            return { ...state, isDirty: true };
        case actionTypes.REMOVE_DIRTY:
            return { ...state, isDirty: false };
        case actionTypes.SET_CHATFLOW:
            return { ...state, chatflow };
        default:
            return state;
    }
};

export default canvasReducer;

// MyComponent.js

import { connect } from 'react-redux';
import { setDirty, removeDirty, setChatFlow } from '../actions';

const MyComponent = ({ isDirty, chatflow, setDirty, removeDirty, setChatFlow }) => {
  // component logic here
}

const mapStateToProps = state => ({
  isDirty: state.canvasReducer.isDirty,
  chatflow: state.canvasReducer.chatflow,
});

const mapDispatchToProps = {
  setDirty,
  removeDirty,
  setChatFlow,
};

export default connect(mapStateToProps, mapDispatchToProps)(MyComponent);

