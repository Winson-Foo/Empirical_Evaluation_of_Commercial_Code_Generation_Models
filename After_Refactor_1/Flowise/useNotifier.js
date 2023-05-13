import React, { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useSnackbar } from 'notistack'
import { removeSnackbar } from 'store/actions'

const useNotifier = () => {
  const dispatch = useDispatch()
  const notifier = useSelector((state) => state.notifier)
  const { notifications } = notifier
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()

  const displaySnackbar = ({ key, message, options }) => {
    enqueueSnackbar(message, {
      key,
      ...options,
      onClose: options.onClose ? options.onClose : undefined,
      onExited: () => {
        dispatch(removeSnackbar(key))
      },
    })
  }

  useEffect(() => {
    notifications.forEach(({ key, message, options = {}, dismissed = false }) => {
      if (dismissed) {
        closeSnackbar(key)
        return
      }

      if (!displayed.includes(key)) {
        displaySnackbar({ key, message, options })
        storeSnackbar(key)
      }
    })
  }, [notifications])

  const storeSnackbar = (id) => {
    displayed = [...displayed, id]
  }

  const removeSnackbarFromStore = (id) => {
    displayed = [...displayed.filter((key) => id !== key)]
  }

  let displayed = []
}

export default useNotifier