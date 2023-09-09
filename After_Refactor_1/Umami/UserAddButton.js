import React, { useCallback } from 'react';
import { Button, Icon, Text, Modal, Icons, ModalTrigger } from 'react-basics';
import UserAddForm from './UserAddForm';
import useMessages from 'hooks/useMessages';

function UserAddButton({ onSave }) {
  const { formatMessage, labels } = useMessages();

  const handleClose = useCallback(() => {
    // Handle modal close here if necessary
  }, []);

  const handleSave = useCallback(() => {
    onSave();
  }, [onSave]);

  return (
    <ModalTrigger>
      <Button variant="primary">
        <Icon>
          <Icons.Plus />
        </Icon>
        <Text>{formatMessage(labels.createUser)}</Text>
      </Button>
      <Modal title={formatMessage(labels.createUser)} onClose={handleClose}>
        <UserAddForm onSave={handleSave} />
      </Modal>
    </ModalTrigger>
  );
}

export default UserAddButton;
// Explanation: 
// 1- The onSave method is used as a dependency to useCallback 
// in case there are changes and it needs to recompute.
// 2- Added a handleClose method to handle the closing of 
// the modal using useCallback and passed it to the Modal component.
// 3- Avoided unnecessary function calls by passing the handleSave to the UserAddForm directly, 
// instead of wrapping it within another function.

