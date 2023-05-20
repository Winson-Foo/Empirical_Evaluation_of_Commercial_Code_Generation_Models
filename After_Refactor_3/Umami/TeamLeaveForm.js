import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, Form, FormButtons, SubmitButton } from 'react-basics';
import { useDeleteMutation } from 'hooks/useApi';
import { useMessages } from 'hooks/useMessages';
import { ConfirmationMessage } from './ConfirmationMessage';
import { ErrorMessage } from './ErrorMessage';

export function TeamLeaveForm({ teamId, userId, teamName, onSave, onClose }) {
  const { formatMessage, messages } = useMessages();

  const { mutate, error, isLoading } = useDeleteMutation(`/team/${teamId}/users/${userId}`);

  const handleSubmit = async () => {
    try {
      await mutate();
      onSave();
      onClose();
    } catch (error) {
      // Error handling can be done here or in a separate component
    }
  };

  return (
    <Form onSubmit={handleSubmit} error={error && <ErrorMessage error={error} />}>
      <ConfirmationMessage teamName={teamName} />

      <FormButtons flex>
        <SubmitButton variant="danger" disabled={isLoading}>
          {formatMessage(messages.leave)}
        </SubmitButton>
        <Button onClick={onClose}>{formatMessage(messages.cancel)}</Button>
      </FormButtons>
    </Form>
  );
}

export default TeamLeaveForm;

// ConfirmationMessage.js
import React from 'react';
import { FormattedMessage } from 'react-intl';

export function ConfirmationMessage({ teamName }) {
  return (
    <p>
      <FormattedMessage
        id="confirmLeave"
        values={{
          name: <b>{teamName}</b>,
        }}
      />
    </p>
  );
}

// ErrorMessage.js
import React from 'react';

export function ErrorMessage({ error }) {
  return <div>Error: {error.message}</div>;
}

// useApi.js
import { useMutation } from 'react-query';

export function useDeleteMutation(url, options = {}) {
  return useMutation(() => del(url), options);
}

// Note: del() method needs to be imported from somewhere, assumed it's already handled.

