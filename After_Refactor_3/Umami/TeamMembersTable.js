import useMessages from 'hooks/useMessages';
import useUser from 'hooks/useUser';
import { ROLES } from 'lib/constants';
import TeamMemberRemoveButton from './TeamMemberRemoveButton';
import SettingsTable from 'components/common/SettingsTable';

const getUsername = (row) => {
  const { user: { username } } = row;
  return username;
};

const getRoleLabel = (formatMessage, labels, role) => {
  const key = Object.keys(ROLES).find(key => ROLES[key] === role);
  return formatMessage(labels[key] || labels.unknown);
};

const cellRender = (row, data, columnName, { formatMessage, labels }) => {
  switch(columnName) {
    case 'username': return getUsername(row);
    case 'role': return getRoleLabel(formatMessage, labels, row.role);
    default: return data[columnName];
  }
};

const TeamMembersTable = ({ data = [], onSave, readOnly }) => {
  const { formatMessage, labels } = useMessages();
  const { user } = useUser();
  const columns = [
    { name: 'username', label: formatMessage(labels.username) },
    { name: 'role', label: formatMessage(labels.role) },
    { name: 'action', label: ' ' },
  ];

  return (
    <SettingsTable data={data} columns={columns} cellRender={(row, data, columnName) => cellRender(row, data, columnName, { formatMessage, labels })}>
      {(row) => {
        if (readOnly) {
          return null;
        }
        const isUser = user.id === row?.user?.id;
        const isTeamOwner = row.role === ROLES.teamOwner;
        const isDisabled = isUser || isTeamOwner;
        return (
          <TeamMemberRemoveButton
            teamId={row.teamId}
            userId={row.userId}
            disabled={isDisabled}
            onSave={onSave}
          />
        );
      }}
    </SettingsTable>
  );
};

export default TeamMembersTable;

