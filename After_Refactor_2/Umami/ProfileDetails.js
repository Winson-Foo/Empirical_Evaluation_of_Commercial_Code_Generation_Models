import { Form, FormRow } from 'react-basics';
import PasswordChangeButton from './PasswordChangeButton';
import useCurrentUser from 'hooks/useCurrentUser';
import useSettingsLabels from 'hooks/useSettingsLabels';
import useUserRoles from 'hooks/useUserRoles';
import DateRangeSettings from 'components/pages/settings/profile/DateRangeSettings';
import LanguageSettings from 'components/pages/settings/profile/LanguageSettings';
import ThemeSettings from 'components/pages/settings/profile/ThemeSettings';
import TimezoneSettings from 'components/pages/settings/profile/TimezoneSettings';

export function ProfileDetails() {
  const { currentUser } = useCurrentUser();
  const { settingsLabels } = useSettingsLabels();
  const { userRoles } = useUserRoles();
  const { cloudMode } = useConfig();

  if (!currentUser) {
    return null;
  }

  const { username, role } = currentUser;

  return (
    <Form>
      <FormRow label={settingsLabels.username}>{username}</FormRow>
      <FormRow label={settingsLabels.role}>
        {settingsLabels[userRoles[role]] || settingsLabels.unknown}
      </FormRow>
      {!cloudMode && (
        <FormRow label={settingsLabels.password}>
          <PasswordChangeButton />
        </FormRow>
      )}
      <FormRow label={settingsLabels.defaultDateRange}>
        <DateRangeSettings />
      </FormRow>
      <FormRow label={settingsLabels.language}>
        <LanguageSettings />
      </FormRow>
      <FormRow label={settingsLabels.timezone}>
        <TimezoneSettings />
      </FormRow>
      <FormRow label={settingsLabels.theme}>
        <ThemeSettings />
      </FormRow>
    </Form>
  );
}

export default ProfileDetails;

