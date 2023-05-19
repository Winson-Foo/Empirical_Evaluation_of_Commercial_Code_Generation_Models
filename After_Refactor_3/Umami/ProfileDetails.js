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

import {
  PROFILE_DETAILS_LABELS,
  ROLES,
  CLOUD_MODE_LABEL,
} from './constants';

const UsernameRow = ({ label, username }) => (
  <FormRow label={label}>{username}</FormRow>
);

UsernameRow.propTypes = {
  label: PropTypes.string.isRequired,
  username: PropTypes.string.isRequired,
};

const RoleRow = ({ label, role, labels }) => (
  <FormRow label={label}>
    {labels[role] || labels.unknown}
  </FormRow>
);

RoleRow.propTypes = {
  label: PropTypes.string.isRequired,
  role: PropTypes.oneOf(Object.keys(ROLES)).isRequired,
  labels: PropTypes.object.isRequired,
};

const PasswordRow = ({ label, cloudMode }) => {
  if (cloudMode) return null;

  return (
    <FormRow label={label}>
      <PasswordChangeButton />
    </FormRow>
  );
};

PasswordRow.propTypes = {
  label: PropTypes.string.isRequired,
  cloudMode: PropTypes.bool.isRequired,
};

const ProfileDetails = () => {
  const { user } = useUser();
  const { formatMessage, labels } = useMessages();
  const { cloudMode } = useConfig();

  if (!user) {
    return null;
  }

  const { username, role } = user;

  return (
    <Form>
      <UsernameRow
        label={formatMessage(PROFILE_DETAILS_LABELS.username)}
        username={username}
      />

      <RoleRow
        label={formatMessage(PROFILE_DETAILS_LABELS.role)}
        role={role}
        labels={labels}
      />

      <PasswordRow
        label={formatMessage(PROFILE_DETAILS_LABELS.password)}
        cloudMode={cloudMode}
      />

      <FormRow
        label={formatMessage(PROFILE_DETAILS_LABELS.defaultDateRange)}
      >
        <DateRangeSetting />
      </FormRow>

      <FormRow label={formatMessage(PROFILE_DETAILS_LABELS.language)}>
        <LanguageSetting />
      </FormRow>

      <FormRow label={formatMessage(PROFILE_DETAILS_LABELS.timezone)}>
        <TimezoneSetting />
      </FormRow>

      <FormRow label={formatMessage(PROFILE_DETAILS_LABELS.theme)}>
        <ThemeSetting />
      </FormRow>
    </Form>
  );
};

export default ProfileDetails;

