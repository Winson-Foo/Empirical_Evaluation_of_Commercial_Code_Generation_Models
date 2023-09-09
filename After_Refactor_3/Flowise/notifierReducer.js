import { ENQUEUE_SNACKBAR, CLOSE_SNACKBAR, REMOVE_SNACKBAR } from '../actions';

const initialState = {
  notifications: []
};

const enqueueSnackbar = (state, action) => {
  const { key, ...notification } = action.notification;
  return {
    ...state,
    notifications: [
      ...state.notifications,
      { key, ...notification }
    ]
  };
};

const closeSnackbar = (state, action) => {
  const { dismissAll, key } = action;
  return {
    ...state,
    notifications: state.notifications.map(notification =>
      (dismissAll || notification.key === key)
        ? { ...notification, dismissed: true }
        : notification
    )
  };
};

const removeSnackbar = (state, action) => {
  const { key } = action;
  return {
    ...state,
    notifications: state.notifications.filter(notification => notification.key !== key)
  };
};

const notifierReducer = (state = initialState, action) => {
  switch (action.type) {
    case ENQUEUE_SNACKBAR:
      return enqueueSnackbar(state, action);
    case CLOSE_SNACKBAR:
      return closeSnackbar(state, action);
    case REMOVE_SNACKBAR:
      return removeSnackbar(state, action);
    default:
      return state;
  }
};

export default notifierReducer;

