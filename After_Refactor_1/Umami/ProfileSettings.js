import React from 'react';
import useMessages from 'hooks/useMessages';

function PageHeader({ title }) {
  const { formatMessage, labels } = useMessages();

  return (
    <header>
      <h1>{formatMessage(labels[title])}</h1>
    </header>
  );
}

export default PageHeader;

Component - ProfileDetails.js:

import React from 'react';

function ProfileDetails() {
  return (
    <section>
      {/* content goes here */}
    </section>
  );
}

export default ProfileDetails;

Component - PageContent.js:

import React from 'react';
import ProfileDetails from './ProfileDetails';

function PageContent() {
  return (
    <main>
      <ProfileDetails />
    </main>
  );
}

export default PageContent;

Component - ProfileSettings.js:

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

