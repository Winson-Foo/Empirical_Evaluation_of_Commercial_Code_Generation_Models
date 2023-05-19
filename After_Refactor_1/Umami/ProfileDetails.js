import React from 'react';
import PropTypes from 'prop-types';
import { Form, FormRow } from 'react-basics';
import TimezoneSetting from 'components/pages/settings/profile/TimezoneSetting';
import DateRangeSetting from 'components/pages/settings/profile/DateRangeSetting';
import LanguageSetting from 'components/pages/settings/profile/LanguageSetting';
import ThemeSetting from 'components/pages/settings/profile/ThemeSetting';
import PasswordChangeButton from './PasswordChangeButton';
import useUser from 'hooks/useUser';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';

const labelsPropTypes = {
  username: PropTypes.string.isRequired,
  password: PropTypes.string.isRequired,
  role: PropTypes.string.isRequired,
  defaultDateRange: PropTypes.string.isRequired,
  language: PropTypes.string.isRequired,
  timezone: PropTypes.string.isRequired,
  theme: PropTypes.string.isRequired,
};

function ProfileDetails({ labels }) {
  const { user } = useUser();
  const { formatMessage } = useMessages();
  const { cloudMode } = useConfig();

  if (!user) {
    return null;
  }

  const { username, role } = user;

  const formatLabel = (label) => formatMessage(labels[label]);

  return (
    <Form>
      <FormRow label={formatLabel('username')}>{username}</FormRow>
      <FormRow label={formatLabel('role')}>
        {formatLabel(role) || formatLabel('unknown')}
      </FormRow>
      {!cloudMode && (
        <FormRow label={formatLabel('password')}>
          <PasswordChangeButton />
        </FormRow>
      )}
      <FormRow label={formatLabel('defaultDateRange')}>
        <DateRangeSetting />
      </FormRow>
      <FormRow label={formatLabel('language')}>
        <LanguageSetting />
      </FormRow>
      <FormRow label={formatLabel('timezone')}>
        <TimezoneSetting />
      </FormRow>
      <FormRow label={formatLabel('theme')}>
        <ThemeSetting />
      </FormRow>
    </Form>
  );
}

ProfileDetails.propTypes = {
  labels: PropTypes.shape(labelsPropTypes).isRequired,
};

ProfileDetails.defaultProps = {
  labels: {
    username: 'username',
    password: 'password',
    role: 'role',
    unknown: 'unknown',
    defaultDateRange: 'defaultDateRange',
    language: 'language',
    timezone: 'timezone',
    theme: 'theme',
  },
};

export default ProfileDetails;

