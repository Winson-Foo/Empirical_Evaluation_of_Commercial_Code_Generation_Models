import React from 'react';
import PropTypes from 'prop-types';
import {
  Form, FormRow, FormInput, FormButtons, TextField, Button, SubmitButton,
} from 'react-basics';
import useApi from 'hooks/useApi';
import { DOMAIN_REGEX } from 'lib/constants';
import useMessages from 'hooks/useMessages';

const AddWebsiteForm = ({ onSave, onClose }) => {
  const { formatMessage, labels, messages } = useMessages();
  const { post, useMutation } = useApi();
  const { mutate, error, isLoading } = useMutation(data => post('/websites', data));

  const handleSubmit = async (data) => {
    mutate(data, {
      onSuccess: async () => {
        onSave();
        onClose();
      },
    });
  };

  const NameInput = () => (
    <FormInput
      name="name"
      rules={{ required: formatMessage(labels.required) }}
    >
      <TextField autoComplete="off" />
    </FormInput>
  );

  const DomainInput = () => (
    <FormInput
      name="domain"
      rules={{
        required: formatMessage(labels.required),
        pattern: { value: DOMAIN_REGEX, message: formatMessage(messages.invalidDomain) },
      }}
    >
      <TextField autoComplete="off" />
    </FormInput>
  );

  return (
    <Form onSubmit={handleSubmit} error={error}>
      <FormRow label={formatMessage(labels.name)}>
        <NameInput />
      </FormRow>
      <FormRow label={formatMessage(labels.domain)}>
        <DomainInput />
      </FormRow>
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
};

AddWebsiteForm.propTypes = {
  onSave: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default AddWebsiteForm;
