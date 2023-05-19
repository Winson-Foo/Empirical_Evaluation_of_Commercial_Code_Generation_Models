// UserAddButton.js
import { Button, Icon, Text, ModalTrigger } from 'react-basics';
import UserAddModal from './UserAddModal';

export function UserAddButton() {
  return (
    <ModalTrigger>
      <Button variant="primary">
        <Icon.Plus />
        <Text>Create User</Text>
      </Button>
      <UserAddModal />
    </ModalTrigger>
  );
}

export default UserAddButton;

// UserAddModal.js
import { Modal } from 'react-basics';
import UserAddForm from './UserAddForm';
import useMessages from 'hooks/useMessages';

export function UserAddModal() {
  const { formatMessage, labels } = useMessages();

  const handleSave = () => {
    // TODO: handle user add logic
  };

  return (
    <Modal title={formatMessage(labels.createUser)}>
      {close => <UserAddForm onSave={handleSave} onClose={close} />}
    </Modal>
  );
}

export default UserAddModal;

// UserAddForm.js
import { Form, Input, Button } from 'react-basics';

export function UserAddForm({ onSave, onClose }) {
  return (
    <Form onSubmit={onSave} onCancel={onClose}>
      <Input label="Name" name="name" required />
      <Input label="Email" name="email" type="email" required />
      <Button type="submit">Save</Button>
    </Form>
  );
}

export default UserAddForm;