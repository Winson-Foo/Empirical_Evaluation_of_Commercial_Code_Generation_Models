import React from 'react';
import { useIntl } from 'react-intl';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import ProfileDetails from './ProfileDetails';

function ProfileSettings() {
  const intl = useIntl();

  return (
    <Page>
      <PageHeader title={intl.formatMessage({ id: 'profile' })} />
      <ProfileDetails />
    </Page>
  );
}

export default ProfileSettings;