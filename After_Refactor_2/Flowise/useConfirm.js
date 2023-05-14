import { useContext, useState } from 'react';
import ConfirmContext from 'store/context/ConfirmContext';
import { HIDE_CONFIRM, SHOW_CONFIRM } from 'store/actions';

const useConfirm = () => {
  const [confirmState, dispatch] = useContext(ConfirmContext);
  const [resolveCallback, setResolveCallback] = useState(() => {});

  const closeConfirm = () => {
    dispatch({ type: HIDE_CONFIRM });
  };

  const handleConfirm = (answer) => {
    closeConfirm();
    resolveCallback(answer);
  };

  const confirm = (confirmPayload) => {
    dispatch({ type: SHOW_CONFIRM, payload: confirmPayload });
    return new Promise((resolve) => {
      setResolveCallback(() => resolve);
    });
  };

  return { confirm, handleConfirm, confirmState };
};

export default useConfirm;