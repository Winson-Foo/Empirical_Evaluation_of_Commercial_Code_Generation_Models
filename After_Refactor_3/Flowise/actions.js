// customizationActions.js
export const SET_MENU = 'customization/SET_MENU'
export const MENU_TOGGLE = 'customization/MENU_TOGGLE'
export const MENU_OPEN = 'customization/MENU_OPEN'
export const SET_FONT_FAMILY = 'customization/SET_FONT_FAMILY'
export const SET_BORDER_RADIUS = 'customization/SET_BORDER_RADIUS'
export const SET_LAYOUT = 'customization/SET_LAYOUT '
export const SET_DARK_MODE = 'customization/SET_DARK_MODE'

// canvasActions.js
export const SET_DIRTY = 'canvas/SET_DIRTY'
export const REMOVE_DIRTY = 'canvas/REMOVE_DIRTY'
export const SET_CHATFLOW = 'canvas/SET_CHATFLOW'

// notifierActions.js
export const ENQUEUE_SNACKBAR = 'notifier/ENQUEUE_SNACKBAR'
export const CLOSE_SNACKBAR = 'notifier/CLOSE_SNACKBAR'
export const REMOVE_SNACKBAR = 'notifier/REMOVE_SNACKBAR'

export const enqueueSnackbar = (notification) => {
    const key = notification.options && notification.options.key

    return {
        type: ENQUEUE_SNACKBAR,
        notification: {
            ...notification,
            key: key || new Date().getTime() + Math.random()
        }
    }
}

export const closeSnackbar = (key) => ({
    type: CLOSE_SNACKBAR,
    dismissAll: !key,
    key
})

export const removeSnackbar = (key) => ({
    type: REMOVE_SNACKBAR,
    key
})

// dialogActions.js
export const SHOW_CONFIRM = 'dialog/SHOW_CONFIRM'
export const HIDE_CONFIRM = 'dialog/HIDE_CONFIRM'

