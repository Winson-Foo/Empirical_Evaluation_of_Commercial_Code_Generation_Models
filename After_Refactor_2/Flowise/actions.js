// actions/customizationActions.js
export const SET_MENU = '@customization/SET_MENU'
export const MENU_TOGGLE = '@customization/MENU_TOGGLE'
export const MENU_OPEN = '@customization/MENU_OPEN'
export const SET_FONT_FAMILY = '@customization/SET_FONT_FAMILY'
export const SET_BORDER_RADIUS = '@customization/SET_BORDER_RADIUS'
export const SET_LAYOUT = '@customization/SET_LAYOUT '
export const SET_DARKMODE = '@customization/SET_DARKMODE'

// actions/canvasActions.js
export const SET_DIRTY = '@canvas/SET_DIRTY'
export const REMOVE_DIRTY = '@canvas/REMOVE_DIRTY'
export const SET_CHATFLOW = '@canvas/SET_CHATFLOW'

// actions/notifierActions.js
export const ENQUEUE_SNACKBAR = 'ENQUEUE_SNACKBAR'
export const CLOSE_SNACKBAR = 'CLOSE_SNACKBAR'
export const REMOVE_SNACKBAR = 'REMOVE_SNACKBAR'

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

// actions/dialogActions.js
export const SHOW_CONFIRM = 'SHOW_CONFIRM'
export const HIDE_CONFIRM = 'HIDE_CONFIRM'

// reducers/customizationReducer.js
// implementation not shown for brevity

// reducers/canvasReducer.js
// implementation not shown for brevity

// reducers/notifierReducer.js
// implementation not shown for brevity

// reducers/dialogReducer.js
// implementation not shown for brevity