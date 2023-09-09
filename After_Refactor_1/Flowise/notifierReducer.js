
export const ENQUEUE_SNACKBAR = 'ENQUEUE_SNACKBAR'
export const CLOSE_SNACKBAR = 'CLOSE_SNACKBAR'
export const REMOVE_SNACKBAR = 'REMOVE_SNACKBAR'
```

Actions:
```
export const enqueueSnackbar = (notification) => ({
    type: ENQUEUE_SNACKBAR,
    notification: {
        key: new Date().getTime() + Math.random(),
        ...notification,
    },
})

export const closeSnackbar = (key) => ({
    type: CLOSE_SNACKBAR,
    dismissAll: !key, // dismiss all if no key provided
    key,
})

export const removeSnackbar = (key) => ({
    type: REMOVE_SNACKBAR,
    key,
})
```

Reducer:
```
import { ENQUEUE_SNACKBAR, CLOSE_SNACKBAR, REMOVE_SNACKBAR } from '../actions'

export const initialState = {
    notifications: [],
}

const notifierReducer = (state = initialState, action) => {
    switch (action.type) {
        case ENQUEUE_SNACKBAR:
            return {
                ...state,
                notifications: [...state.notifications, action.notification],
            }

        case CLOSE_SNACKBAR:
            return {
                ...state,
                notifications: state.notifications.map((notification) =>
                    ((action.dismissAll || notification.key === action.key) && !notification.dismissed) ? { 
                        ...notification, 
                        dismissed: true 
                    } : { 
                        ...notification 
                    }
                ),
            }

        case REMOVE_SNACKBAR:
            return {
                ...state,
                notifications: state.notifications.filter((notification) => notification.key !== action.key),
            }

        default:
            return state
    }
}

export default notifierReducer