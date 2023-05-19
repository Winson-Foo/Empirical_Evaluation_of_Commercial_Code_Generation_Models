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