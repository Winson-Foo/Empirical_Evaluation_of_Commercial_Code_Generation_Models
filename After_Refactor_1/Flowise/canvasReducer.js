// action - state management
import { SET_DIRTY, REMOVE_DIRTY, SET_CHATFLOW } from '../actions';

// Define the initial state
const initialState = {
  isDirty: false,
  chatflow: null,
};

// ==============================|| CANVAS REDUCER ||============================== //

const canvasReducer = (state = initialState, action) => {
  // Destructure the action object
  const { type, chatflow } = action;

  switch (type) {
    case SET_DIRTY:
      return { ...state, isDirty: true };
    case REMOVE_DIRTY:
      return { ...state, isDirty: false };
    case SET_CHATFLOW:
      return { ...state, chatflow };
    default:
      return state;
  }
};

export default canvasReducer;

