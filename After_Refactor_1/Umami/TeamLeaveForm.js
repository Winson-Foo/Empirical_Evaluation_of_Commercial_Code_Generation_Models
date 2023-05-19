import React from 'react';
import { FormattedMessage } from 'react-intl';
import FormComponent from './FormComponent';
import FormButtonsComponent from './FormButtonsComponent';
import SubmitButtonComponent from './SubmitButtonComponent';
import ButtonComponent from './ButtonComponent';

import useApi from '../hooks/useApi';
import useMessages from '../hooks/useMessages';

const TeamLeaveForm = ({ teamId, userId, teamName, onSave, onClose }) => {
  const { formatMessage, labels, messages } = useMessages();
  const { del, useMutation } = useApi();

  const { mutate, error, isLoading } = useMutation(() => del(`/team/${teamId}/users/${userId}`));

  const handleSubmit = async () => {
    mutate(
      {},
      {
        onSuccess: async () => {
          onSave();
          onClose();
        },
      }
    );
  };

  return (
    <FormComponent onSubmit={handleSubmit} error={error}>
      <p>
        <FormattedMessage {...messages.confirmLeave} values={{ name: <b>{teamName}</b> }} />
      </p>
      <FormButtonsComponent flex>
        <SubmitButtonComponent variant="danger" disabled={isLoading}>
          {formatMessage(labels.leave)}
        </SubmitButtonComponent>
        <ButtonComponent onClick={onClose}>{formatMessage(labels.cancel)}</ButtonComponent>
      </FormButtonsComponent>
    </FormComponent>
  );
};

export default TeamLeaveForm;

