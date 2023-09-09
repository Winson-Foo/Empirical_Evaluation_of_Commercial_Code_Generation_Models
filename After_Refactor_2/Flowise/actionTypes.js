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