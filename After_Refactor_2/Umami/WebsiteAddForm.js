import React from 'react';
import {
  Form,
  FormRow,
  FormInput,
  FormButtons,
  TextField,
  Button,
  SubmitButton,
} from 'react-basics';

import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

import { DOMAIN_REGEX } from 'lib/constants';

// Define the form component
function WebsiteAddForm({ onSave, onClose }) {

  // Initialize hooks
  const { formatMessage, labels, messages } = useMessages();
  const { post, useMutation } = useApi();
  const { mutate, error, isLoading } = useMutation(data => post('/websites', data));

  // Handle form submission
  const handleSubmit = async data => {
    mutate(data, {
      onSuccess: async () => {
        onSave();
        onClose();
      },
    });
  };

  // Render the form component
  return (
    <Form onSubmit={handleSubmit} error={error}>
      {/* Name input */}
      <FormRow label={formatMessage(labels.name)}>
        <FormInput name="name" rules={{ required: formatMessage(labels.required) }}>
          <TextField autoComplete="off" />
        </FormInput>
      </FormRow>

      {/* Domain input */}
      <FormRow label={formatMessage(labels.domain)}>
        <FormInput
          name="domain"
          rules={{
            required: formatMessage(labels.required),
            pattern: { value: DOMAIN_REGEX, message: formatMessage(messages.invalidDomain) },
          }}
        >
          <TextField autoComplete="off" />
        </FormInput>
      </FormRow>

      {/* Submit and cancel buttons */}
      <FormButtons flex>
        <SubmitButton variant="primary" disabled={false}>
          {formatMessage(labels.save)}
        </SubmitButton>
        <Button disabled={isLoading} onClick={onClose}>
          {formatMessage(labels.cancel)}
        </Button>
      </FormButtons>
    </Form>
  );
}

export default WebsiteAddForm;

