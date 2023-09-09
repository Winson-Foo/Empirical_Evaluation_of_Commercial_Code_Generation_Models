// confirmReducer.js
export const initialState = {
    isOpen: false,
    title: '',
    message: '',
    confirmLabel: 'Confirm',
    cancelLabel: 'Cancel',
    onConfirm: () => {},
    onCancel: () => {},
  }
  
  const confirmReducer = (state, action) => {
    switch (action.type) {
      case 'OPEN_CONFIRM_DIALOG':
        return {
          ...state,
          isOpen: true,
          title: action.payload.title,
          message: action.payload.message,
          confirmLabel: action.payload.confirmLabel,
          cancelLabel: action.payload.cancelLabel,
          onConfirm: action.payload.onConfirm,
          onCancel: action.payload.onCancel,
        }
      case 'CLOSE_CONFIRM_DIALOG':
        return {
          ...state,
          isOpen: false,
        }
      default:
        return state
    }
  }
  
  export default confirmReducer