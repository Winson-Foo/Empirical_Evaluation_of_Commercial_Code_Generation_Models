// ConfirmDialogContainer.jsx
import React from 'react';
import { createPortal } from 'react-dom';
import useConfirm from 'hooks/useConfirm';
import ConfirmDialog from './ConfirmDialog';
import { CONFIRM_DIALOG_STRINGS } from './constants';

const ConfirmDialogContainer = () => {
  const { onConfirm, onCancel, confirmState } = useConfirm();
  const portalElement = document.getElementById('portal');

  const component = <ConfirmDialog {...confirmState} onCancel={onCancel} onConfirm={onConfirm} />;

  return createPortal(component, portalElement);
};

export default ConfirmDialogContainer;