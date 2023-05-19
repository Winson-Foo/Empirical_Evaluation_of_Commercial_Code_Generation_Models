import React from 'react';
import PropTypes from 'prop-types';
import Link from 'next/link';
import { Button, Icon, Icons, Text } from 'react-basics';
import SettingsTable from 'components/common/SettingsTable';
import TeamWebsiteRemoveButton from './TeamWebsiteRemoveButton';
import useConfig from 'hooks/useConfig';
import useMessages from 'hooks/useMessages';
import useUser from 'hooks/useUser';

function TeamWebsites({ data = [], onSave }) {
  const { formatMessage, labels } = useMessages();
  const { openExternal } = useConfig();
  const { user } = useUser();

  const columns = [
    { name: 'name', label: formatMessage(labels.name) },
    { name: 'domain', label: formatMessage(labels.domain) },
    { name: 'action', label: ' ' },
  ];

  const renderRow = ({ teamId, website }) => {
    const { id: websiteId, name, domain, userId } = website;
    const { teamUser } = team;
    const owner = teamUser[0];
    const canRemove = user.id === userId || user.id === owner.userId;

    return (
      <>
        <Link href={`/websites/${websiteId}`} target={openExternal ? '_blank' : null}>
          <Button>
            <Icon>
              <Icons.External />
            </Icon>
            <Text>{formatMessage(labels.view)}</Text>
          </Button>
        </Link>
        {canRemove && (
          <TeamWebsiteRemoveButton
            teamId={teamId}
            websiteId={websiteId}
            onSave={onSave}
          ></TeamWebsiteRemoveButton>
        )}
      </>
    );
  };

  return <SettingsTable columns={columns} data={data} renderRow={renderRow} />;
}

TeamWebsites.propTypes = {
  data: PropTypes.array,
  onSave: PropTypes.func,
};

export default TeamWebsites;

