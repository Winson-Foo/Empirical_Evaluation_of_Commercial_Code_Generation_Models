import React from 'react';
import PropTypes from 'prop-types';
import Link from 'next/link';
import { Button, Icon, Icons, Text } from 'react-basics';

import useMessages from 'hooks/useMessages';
import useUser from 'hooks/useUser';
import useConfig from 'hooks/useConfig';

import TeamWebsiteRemoveButton from './TeamWebsiteRemoveButton';
import SettingsTable from 'components/common/SettingsTable';

function canRemoveWebsite(user, websiteUserId, teamOwnerUserId) {
  return user.id === websiteUserId || user.id === teamOwnerUserId;
}

function TeamWebsitesTableRow(props) {
  const { website, team, onSave, openExternal, formatMessage, labels, user } = props;
  const { id: websiteId, name, domain, userId } = website;
  const { teamId, teamUser } = team;
  const teamOwner = teamUser[0];
  const canRemove = canRemoveWebsite(user, userId, teamOwner.userId);

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
        <TeamWebsiteRemoveButton teamId={teamId} websiteId={websiteId} onSave={onSave} />
      )}
    </>
  );
}

function TeamWebsitesTable(props) {
  const { data, onSave } = props;
  const { formatMessage, labels } = useMessages();
  const { openExternal } = useConfig();
  const { user } = useUser();
  const columns = [
    { name: 'name', label: formatMessage(labels.name) },
    { name: 'domain', label: formatMessage(labels.domain) },
    { name: 'action', label: ' ' },
  ];

  return (
    <SettingsTable columns={columns} data={data}>
      {(row) => {
        return (
          <TeamWebsitesTableRow
            website={row.website}
            team={row.team}
            onSave={onSave}
            openExternal={openExternal}
            formatMessage={formatMessage}
            labels={labels}
            user={user}
          />
        );
      }}
    </SettingsTable>
  );
}

TeamWebsitesTableRow.propTypes = {
  website: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    domain: PropTypes.string.isRequired,
    userId: PropTypes.string.isRequired,
  }).isRequired,
  team: PropTypes.shape({
    teamId: PropTypes.string.isRequired,
    teamUser: PropTypes.arrayOf(
      PropTypes.shape({
        userId: PropTypes.string.isRequired,
        role: PropTypes.string.isRequired,
      })
    ).isRequired,
  }).isRequired,
  onSave: PropTypes.func.isRequired,
  openExternal: PropTypes.bool.isRequired,
  formatMessage: PropTypes.func.isRequired,
  labels: PropTypes.object.isRequired,
  user: PropTypes.shape({
    id: PropTypes.string.isRequired,
  }).isRequired,
};

TeamWebsitesTable.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      website: TeamWebsitesTableRow.propTypes.website.isRequired,
      team: TeamWebsitesTableRow.propTypes.team.isRequired,
    })
  ).isRequired,
  onSave: PropTypes.func.isRequired,
};

export default TeamWebsitesTable;

