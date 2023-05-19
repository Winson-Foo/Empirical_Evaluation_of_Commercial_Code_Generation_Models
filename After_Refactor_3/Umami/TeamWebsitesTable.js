import useMessages from 'hooks/useMessages';
import useUser from 'hooks/useUser';
import Link from 'next/link';
import { Button, Icon, Icons, Text } from 'react-basics';
import TeamWebsiteRemoveButton from './TeamWebsiteRemoveButton';
import SettingsTable from 'components/common/SettingsTable';
import useConfig from 'hooks/useConfig';

const TeamWebsiteTableRow = ({ row, user, formatMessage, labels, openExternal, onSave }) => {
  const { teamId } = row;
  const { id: websiteId, name, domain, userId } = row.website;
  const { teamUser } = row.team;
  const owner = teamUser[0];
  const canRemove = user.id === userId || user.id === owner.userId;

  const onViewWebsiteClick = () => {
    if (openExternal) {
      window.open(`/websites/${websiteId}`);
    }
  };

  return (
    <>
      <Button onClick={onViewWebsiteClick}>
        <Icon>
          <Icons.External />
        </Icon>
        <Text>{formatMessage(labels.view)}</Text>
      </Button>
      {canRemove && (
        <TeamWebsiteRemoveButton teamId={teamId} websiteId={websiteId} onSave={onSave} />
      )}
    </>
  );
};

const TeamWebsitesTable = ({ data = [] }) => {
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
      {(row) => (
        <TeamWebsiteTableRow
          row={row}
          user={user}
          formatMessage={formatMessage}
          labels={labels}
          openExternal={openExternal}
          onSave={onSave}
        />
      )}
    </SettingsTable>
  );
};

export default TeamWebsitesTable;

