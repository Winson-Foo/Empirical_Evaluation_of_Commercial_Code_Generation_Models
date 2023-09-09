import React from 'react';
import PropTypes from 'prop-types';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import ProfileDetails from './ProfileDetails';
import useMessages from 'hooks/useMessages';

function ProfileSettings(props) {
  const { formatMessage, labels } = useMessages();

  return (
    <Page>
      <PageHeader title={formatMessage(labels.profile)} />
      <ProfileDetails {...props} />
    </Page>
  );
}

ProfileSettings.propTypes = {
  // Define props here, if any
};

export default ProfileSettings;