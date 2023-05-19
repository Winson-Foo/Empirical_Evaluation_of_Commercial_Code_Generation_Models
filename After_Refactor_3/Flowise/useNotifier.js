import React, { useState, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useSnackbar } from 'notistack'
import { removeSnackbar } from 'store/actions'

const useNotifier = () => {
  const dispatch = useDispatch()
  const notifier = useSelector((state) => state.notifier)
  const { notifications } = notifier

  const [displayed, setDisplayed] = useState([])

  const { enqueueSnackbar, closeSnackbar } = useSnackbar()

  const displayNotification = ({ key, message, options = {}, dismissed = false }) => {
    if (dismissed) {
      closeSnackbar(key)
      return
    }

    if (!displayed.includes(key)) {
      const onClose = (event, reason, myKey) => {
        if (options.onClose) {
          options.onClose(event, reason, myKey)
        }
      }

      const onExited = (event, myKey) => {
        dispatch(removeSnackbar(myKey))
        setDisplayed((prevDisplayed) => prevDisplayed.filter((displayedKey) => displayedKey !== myKey))
      }

      enqueueSnackbar(message, {
        key,
        ...options,
        onClose,
        onExited
      })

      setDisplayed((prevDisplayed) => [...prevDisplayed, key])
    }
  }

  useEffect(() => {
    notifications.forEach((notification) => {
      displayNotification(notification)
    })
  }, [notifications, displayed, enqueueSnackbar, closeSnackbar, dispatch])

  return null
}

export default useNotifier