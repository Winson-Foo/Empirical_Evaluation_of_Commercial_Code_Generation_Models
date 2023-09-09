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
import { DOMAIN_REGEX } from 'lib/constants';
import useMessages from 'hooks/useMessages';

// Reusable function to validate form input
const validateFormInput = (value, label, required, pattern, message) => {
  const rules = { required: required && `${label} is required` };

  if (pattern) {
    rules.pattern = { value: pattern, message };
  }

  return (
    <FormInput name={label} rules={rules}>
      <TextField autoComplete="off" value={value} />
    </FormInput>
  );
};

const WebsiteAddForm = ({ onSave, onClose }) => {
  // Custom hook to handle API calls
  const { post, useMutation } = useApi();
  const { mutate, error, isLoading } = useMutation(data =>
    post('/websites', data)
  );

  const { formatMessage, labels, messages } = useMessages();

  const handleSubmit = async data => {
    // Submit form data to API
    mutate(data, {
      onSuccess: async () => {
        onSave();
        onClose();
      },
    });
  };

  return (
    <Form onSubmit={handleSubmit} error={error}>
      {/* Form input for website name */}
      <FormRow label={formatMessage(labels.name)}>
        {validateFormInput(
          '',
          'name',
          true,
          null,
          formatMessage(labels.required)
        )}
      </FormRow>

      {/* Form input for website domain */}
      <FormRow label={formatMessage(labels.domain)}>
        {validateFormInput(
          '',
          'domain',
          true,
          DOMAIN_REGEX,
          formatMessage(messages.invalidDomain)
        )}
      </FormRow>

      {/* Form buttons */}
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

export default WebsiteAddForm;

