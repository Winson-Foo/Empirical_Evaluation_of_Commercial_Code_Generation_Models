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