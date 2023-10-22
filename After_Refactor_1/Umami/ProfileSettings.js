import React from 'react';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import PageContent from './PageContent';

function ProfileSettings() {
  return (
    <Page>
      <PageHeader title="profile" />
      <PageContent />
    </Page>
  );
}

export default ProfileSettings;