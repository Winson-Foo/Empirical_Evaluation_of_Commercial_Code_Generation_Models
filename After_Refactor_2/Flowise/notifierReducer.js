import { ENQUEUE_SNACKBAR, CLOSE_SNACKBAR, REMOVE_SNACKBAR } from '../actions'

export const initialState = {
  notifications: [],
}

const mapNotifications = (notifications, action) => {
  return notifications.reduce((acc, notification) => {
    if (action.dismissAll || notification.key === action.key) {
      acc.push({ ...notification, dismissed: true })
    } else {
      acc.push(notification)
    }
    return acc
  }, [])
}

const notifierReducer = (state = initialState, action) => {
  switch (action.type) {
    case ENQUEUE_SNACKBAR:
      return {
        ...state,
        notifications: [
          ...state.notifications,
          {
            key: action.key,
            ...action.notification,
          },
        ],
      }

    case CLOSE_SNACKBAR:
      return {
        ...state,
        notifications: mapNotifications(state.notifications, action),
      }

    case REMOVE_SNACKBAR:
      return {
        ...state,
        notifications: state.notifications.filter(
          (notification) => notification.key !== action.key
        ),
      }

    default:
      return state
  }
}

export default notifierReducer