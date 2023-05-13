export const SHOW_CONFIRM = 'SHOW_CONFIRM';
export const HIDE_CONFIRM = 'HIDE_CONFIRM';

import { SHOW_CONFIRM, HIDE_CONFIRM } from '../actions';

const initialState = {
  show: false,
  title: '',
  description: '',
  confirmButtonName: 'OK',
  cancelButtonName: 'Cancel',
};

const alertReducer = (state = initialState, action) => {
  switch (action.type) {
    case SHOW_CONFIRM: {
      const { title, description, confirmButtonName, cancelButtonName } = action.payload;
      return {
        ...state,
        show: true,
        title,
        description,
        confirmButtonName,
        cancelButtonName,
      };
    }
    case HIDE_CONFIRM:
      return initialState;
    default:
      return state;
  }
};

export default alertReducer;