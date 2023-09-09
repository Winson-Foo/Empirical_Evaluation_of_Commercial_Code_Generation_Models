import { SHOW_CONFIRM, HIDE_CONFIRM } from '../actions';

const initialState = {
  show: false,
  title: '',
  description: '',
  confirmButtonName: 'OK',
  cancelButtonName: 'Cancel',
};

const alertReducer = (state = initialState, action) => {
  const { type, payload } = action;

  switch (type) {
    case SHOW_CONFIRM:
      return {
        ...state,
        show: true,
        ...payload,
      };
    case HIDE_CONFIRM:
      return initialState;
    default:
      return state;
  }
};

export default alertReducer;

