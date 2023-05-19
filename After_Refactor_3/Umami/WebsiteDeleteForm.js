import React from 'react';
import PropTypes from 'prop-types';
import { Button, Form, FormRow, FormButtons, FormInput, SubmitButton, TextField } from 'react-basics';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

const CONFIRM_VALUE = 'DELETE';

function WebsiteDeleteForm({ websiteId, onSave, onClose }) {
  // 1. Use more descriptive variable names
  const { formatMessage, labels, messages, FormattedMessage } = useMessages();
  const { del, useMutation } = useApi();

  // Group related code together with comments
  // -----------------------------------------

  // Define mutation
  const { mutate, error } = useMutation(data => del(`/websites/${websiteId}`, data));

  // Handle form submit
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
        {/* 1. Use more descriptive messages */}
        <FormattedMessage
          {...messages.deleteWebsite}
          values={{ confirmation: <b>{CONFIRM_VALUE}</b> }}
        />
      </p>
      <FormRow label={formatMessage(labels.confirm)}>
        <FormInput name="confirmation" rules={{ validate: value => value === CONFIRM_VALUE }}>
          <TextField autoComplete="off" />
        </FormInput>
      </FormRow>
      <FormButtons flex>
        <SubmitButton variant="danger">{formatMessage(labels.delete)}</SubmitButton>
        <Button onClick={onClose}>{formatMessage(labels.cancel)}</Button>
      </FormButtons>
    </Form>
  );
}

WebsiteDeleteForm.propTypes = {
  websiteId: PropTypes.string.isRequired,
  onSave: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default WebsiteDeleteForm;

