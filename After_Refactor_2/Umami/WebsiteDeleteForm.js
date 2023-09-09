import { Button, Form, FormRow, FormButtons, FormInput, SubmitButton } from 'react-basics';
import useDeleteApi from 'hooks/useDeleteApi';
import useMessages from 'hooks/useMessages';

const DELETE_CONFIRMATION_VALUE = 'DELETE';

export function WebsiteDeleteForm({ websiteId, onSave, onClose }) {
  const { labels, messages, formatMessage } = useMessages();
  const { deleteWebsite, error } = useDeleteApi(`/websites/${websiteId}`);

  const handleDelete = async () => {
    await deleteWebsite();
    onSave();
    onClose();
  };

  const handleSubmit = async ({ confirmation }) => {
    if (confirmation === DELETE_CONFIRMATION_VALUE) {
      await handleDelete();
    }
  };

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <p>{formatMessage(messages.deleteWebsite, { confirmation: <b>{DELETE_CONFIRMATION_VALUE}</b> })}</p>
      <FormRow label={formatMessage(labels.confirm)}>
        <FormInput name="confirmation" rules={{ validate: value => value === DELETE_CONFIRMATION_VALUE }}>
          <input autoComplete="off" />
        </FormInput>
      </FormRow>
      <FormButtons flex>
        <SubmitButton variant="danger">{formatMessage(labels.delete)}</SubmitButton>
        <Button onClick={onClose}>{formatMessage(labels.cancel)}</Button>
      </FormButtons>
    </Form>
  );
}

export default WebsiteDeleteForm;

