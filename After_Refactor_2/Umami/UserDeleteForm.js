import { useMutation } from '@tanstack/react-query';
import { Button, Form, FormButtons, SubmitButton } from 'react-basics';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

function deleteUser(userId, onSuccess) {
  const { del } = useApi();
  const { mutate } = useMutation(() => del(`/users/${userId}`), {
    onSuccess: async () => {
      onSuccess();
    },
  });

  return mutate;
}

export function UserDeleteForm({ userId, username, onSave, onClose }) {
  const { formatMessage, FormattedMessage, labels, messages } = useMessages();
  const [mutate, { error, isLoading }] = useMutation(deleteUser(userId, onSave));

  const handleSubmit = async (data) => {
    mutate(data);
  };

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <p>
        <FormattedMessage {...messages.confirmDelete} values={{ target: <b>{username}</b> }} />
      </p>
      <<FormButtons flex>
        <SubmitButton variant="danger" disabled={isLoading}>
          {formatMessage(labels.delete)}
        </SubmitButton>
        <Button disabled={isLoading} onClick={onClose}>
          {formatMessage(labels.cancel)}
        </Button>
      </FormButtons>
    </Form>
  );
}

export default UserDeleteForm;