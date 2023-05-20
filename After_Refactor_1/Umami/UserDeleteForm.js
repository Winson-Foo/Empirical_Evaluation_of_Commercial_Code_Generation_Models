// Import only what's needed
import { useMutation } from '@tanstack/react-query';
import { Form, FormButtons, SubmitButton } from 'react-basics';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

// Define constants for better readability
const DELETE_USER_ENDPOINT = '/users/'
const SUCCESS_CALLBACK = async () => {
  onSave();
  onClose();
};

function UserDeleteForm({ userId, username, onSave, onClose }) {
  // Destructure the properties for better readability
  const { formatMessage, FormattedMessage, labels, messages } = useMessages();
  const { del } = useApi();
  const { mutate, error, isLoading } = useMutation(() => del(`${DELETE_USER_ENDPOINT}${userId}`), { onSuccess: SUCCESS_CALLBACK });

  const handleSubmit = async data => {
    mutate(data);
  };

  const deleteButton = (
    <SubmitButton variant="danger" disabled={isLoading}>
      {formatMessage(labels.delete)}
    </SubmitButton>
  );

  const cancelButton = (
    <Button disabled={isLoading} onClick={onClose}>
      {formatMessage(labels.cancel)}
    </Button>
  );

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <p>
        <FormattedMessage {...messages.confirmDelete} values={{ target: <b>{username}</b> }} />
      </p>
      <FormButtons flex>
        {deleteButton}
        {cancelButton}
      </FormButtons>
    </Form>
  );
}

export default UserDeleteForm;