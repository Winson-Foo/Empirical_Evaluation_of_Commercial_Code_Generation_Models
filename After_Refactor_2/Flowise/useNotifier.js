import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useSnackbar } from 'notistack';
import { removeSnackbar } from 'store/actions';

const NOTIFICATION_DISPLAY_TIME = 4000;
const NOTIFICATION_DISMISS_TIME = 2000;

const useNotifier = () => {
  const dispatch = useDispatch();
  const { notifications } = useSelector(state => state.notifier);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();

  const storeDisplayedNotification = notificationId => {
    dispatch(setDisplayedNotification(notificationId));
  };

  const removeDisplayedNotification = notificationId => {
    dispatch(removeDisplayedNotification(notificationId));
  };

  const handleOnClose = (notificationId, options) => (
    event,
    reason,
    myKey
  ) => {
    if (options.onClose) {
      options.onClose(event, reason, myKey);
    }
    removeDisplayedNotification(notificationId);
    closeSnackbar(notificationId);
  };

  const handleOnExited = notificationId => () =>
    dispatch(removeSnackbar(notificationId));

  const displayNotification = notification => {
    const {
      id,
      message,
      options = {},
      dismissed = false,
    } = notification;
    if (dismissed) {
      closeSnackbar(id);
      return;
    }
    if (notifications.displayed.includes(id)) return;

    enqueueSnackbar(message, {
      key: id,
      autoHideDuration: NOTIFICATION_DISPLAY_TIME,
      ...options,
      onClose: handleOnClose(id, options),
      onExited: handleOnExited(id),
    });

    storeDisplayedNotification(id);

    setTimeout(() => {
      removeDisplayedNotification(id);
      closeSnackbar(id);
    }, NOTIFICATION_DISPLAY_TIME + NOTIFICATION_DISMISS_TIME);
  };

  useEffect(() => {
    notifications.queue.forEach(displayNotification);
  }, [notifications.queue]);

  return null;
};

export default useNotifier;

