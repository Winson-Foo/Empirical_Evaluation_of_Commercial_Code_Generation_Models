import {
  Button,
  Form,
  FormRow,
  FormButtons,
  FormInput,
  SubmitButton,
  TextField,
} from 'react-basics';

import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

// Constants
const CONFIRMATION_VALUE = 'DELETE';

// Strings
const DELETE_LABEL = 'delete';
const CANCEL_LABEL = 'cancel';
const CONFIRM_LABEL = 'confirm';
const DELETE_WEBSITE_MESSAGE_ID = 'deleteWebsite';

/**
 * Form component that allows the user to delete a website.
 */
export function WebsiteDeleteForm({ websiteId, onSave, onClose }) {
  // Hooks
  const { formatMessage, labels, messages, FormattedMessage } = useMessages();
  const { del, useMutation } = useApi();
  const { mutate, error } = useMutation(data => del(`/websites/${websiteId}`, data));

  // Submit handler
  const handleSubmit = async data => {
    mutate(data, {
      onSuccess: async () => {
        onSave();
        onClose();
      },
    });
  };

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <p>
        <FormattedMessage
          {...messages[DELETE_WEBSITE_MESSAGE_ID]}
          values={{ confirmation: <b>{CONFIRMATION_VALUE}</b> }}
        />
      </p>
      <ConfirmationInput />
      <FormButtons flex>
        <DeleteButton />
        <CancelButton />
      </FormButtons>
    </Form>
  );
}

/**
 * Input component that requires the user to confirm the deletion by typing a confirmation value.
 */
function ConfirmationInput() {
  return (
    <FormRow label={formatMessage(labels[CONFIRM_LABEL])}>
      <FormInput
        name="confirmation"
        rules={{ validate: value => value === CONFIRMATION_VALUE }}
      >
        <TextField autoComplete="off" />
      </FormInput>
    </FormRow>
  );
}

/**
 * Button component that triggers the deletion of the website when clicked.
 */
function DeleteButton() {
  // Hooks
  const { formatMessage, labels } = useMessages();

  return (
    <SubmitButton variant="danger">
      {formatMessage(labels[DELETE_LABEL])}
    </SubmitButton>
  );
}

/**
 * Button component that cancels the deletion and closes the form when clicked.
 */
function CancelButton() {
  // Hooks
  const { formatMessage, labels } = useMessages();

  return (
    <Button onClick={onClose}>
      {formatMessage(labels[CANCEL_LABEL])}
    </Button>
  );
}

export default WebsiteDeleteForm;

