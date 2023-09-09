import React from 'react';
import { useMutation } from '@tanstack/react-query';
import PropTypes from 'prop-types';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';
import { Button, Form, FormButtons, SubmitButton } from 'react-basics';

const UserDeleteForm = ({ userId, username, onSave, onClose }) => {
  const { formatMessage, FormattedMessage } = useMessages();
  const { del } = useApi();
  const { mutate, error, isLoading } = useMutation(() => del(`/users/${userId}`));

  const handleSubmit = async data => {
    mutate(data, {
      onSuccess: async () => {
        onSave();
        onClose();
      },
    });
  };

  const formatDeleteMessage = () => (
    <FormattedMessage
      id="confirmDelete"
      defaultMessage="Are you sure you want to delete {target}?"
      values={{ target: <b>{username}</b> }}
    />
  );

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <p>{formatDeleteMessage()}</p>
      <FormButtons flex>
        <SubmitButton variant="danger" disabled={isLoading}>
          {formatMessage('delete')}
        </SubmitButton>
        <Button disabled={isLoading} onClick={onClose}>
          {formatMessage('cancel')}
        </Button>
      </FormButtons>
    </Form>
  );
};

UserDeleteForm.propTypes = {
  userId: PropTypes.string.isRequired,
  username: PropTypes.string.isRequired,
  onSave: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default UserDeleteForm;