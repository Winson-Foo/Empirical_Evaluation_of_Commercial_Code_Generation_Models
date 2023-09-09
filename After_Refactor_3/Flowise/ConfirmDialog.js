import { createPortal } from 'react-dom';
import { Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';
import useConfirm from 'hooks/useConfirm';
import { StyledButton } from 'ui-component/button/StyledButton';

const ConfirmationModalContent = ({ title, description, onCancel, onConfirm, cancelButtonName, confirmButtonName }) => (
  <>
    <DialogTitle sx={{ fontSize: '1rem' }}>{title}</DialogTitle>
    <DialogContent>
      <DialogContentText sx={{ color: 'black' }}>{description}</DialogContentText>
    </DialogContent>
    <DialogActions>
      <Button onClick={onCancel}>{cancelButtonName}</Button>
      <StyledButton variant='contained' onClick={onConfirm}>
        {confirmButtonName}
      </StyledButton>
    </DialogActions>
  </>
);

const ConfirmationModal = () => {
  const { onConfirm, onCancel, confirmState: { show, title, description, cancelButtonName, confirmButtonName } } = useConfirm();
  const portalElement = document.getElementById('portal');

  const component = show ? (
    <Dialog
      fullWidth
      maxWidth='xs'
      open={show}
      onClose={onCancel}
      aria-labelledby='alert-dialog-title'
      aria-describedby='alert-dialog-description'
    >
      <ConfirmationModalContent
        title={title}
        description={description}
        onCancel={onCancel}
        onConfirm={onConfirm}
        cancelButtonName={cancelButtonName}
        confirmButtonName={confirmButtonName}
      />
    </Dialog>
  ) : null;

  return createPortal(component, portalElement);
};

export default ConfirmationModal;

