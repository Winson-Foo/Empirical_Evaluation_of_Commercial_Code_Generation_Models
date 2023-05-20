import React, { useState } from 'react';
import { Button, Icon, Text, Modal, ModalTrigger } from 'react-basics';
import UserAddForm from './UserAddForm';
import { useMessages } from 'hooks';

function CreateUserButton({ onSave }) {
  const { formatMessage, labels } = useMessages();
  const [showModal, setShowModal] = useState(false);

  const handleSave = () => {
    onSave();
    setShowModal(false);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const openModal = () => {
    setShowModal(true);
  };

  return (
    <>
      <Button variant="primary" onClick={openModal}>
        <Icon name="Plus" />
        <Text>{formatMessage(labels.createUser)}</Text>
      </Button>
      <Modal isOpen={showModal} title={formatMessage(labels.createUser)}>
        <UserAddForm onSave={handleSave} onClose={closeModal} />
      </Modal>
    </>
  );
}

export default CreateUserButton;