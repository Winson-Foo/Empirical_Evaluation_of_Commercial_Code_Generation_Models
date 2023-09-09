import { SHOW_CONFIRM, HIDE_CONFIRM } from '../actions'

export const initialState = {
  isConfirmVisible: false,
  confirmTitle: '',
  confirmDescription: '',
  confirmButtonName: 'OK',
  cancelButtonName: 'Cancel'
}

const alertReducer = (state = initialState, action) => {
  switch (action.type) {
    case SHOW_CONFIRM:
      const { title, description, confirmButtonName, cancelButtonName } = action.payload
      return {
        ...state,
        isConfirmVisible: true,
        confirmTitle: title,
        confirmDescription: description,
        confirmButtonName: confirmButtonName,
        cancelButtonName: cancelButtonName
      }
    case HIDE_CONFIRM:
      return {
        ...state,
        isConfirmVisible: false
      }
    default:
      return state
  }
}

export default alertReducer