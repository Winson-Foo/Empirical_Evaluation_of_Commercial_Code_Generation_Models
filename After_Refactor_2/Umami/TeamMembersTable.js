import React from 'react';
import SettingsTable from 'components/common/SettingsTable';
import TeamMemberRemoveButton from './TeamMemberRemoveButton';

const COLUMN_NAMES = {
  USERNAME: 'username',
  ROLE: 'role',
  ACTION: 'action',
};

const ROLES = {
  TEAM_OWNER: 'teamOwner',
  MEMBER: 'member',
  UNKNOWN: 'unknown',
};

function formatRoleLabel(role, labels) {
  const roleKey = Object.keys(ROLES).find((key) => ROLES[key] === role) || ROLES.UNKNOWN;
  return labels[roleKey];
}

function TeamMembersTable({ teamMembers = [], onSave, readOnly }) {
  return (
    <SettingsTable
      data={teamMembers}
      columns={[
        { name: COLUMN_NAMES.USERNAME, label: 'Username' },
        { name: COLUMN_NAMES.ROLE, label: 'Role' },
        { name: COLUMN_NAMES.ACTION, label: ' ' },
      ]}
      cellRender={(row, data, columnKey) => {
        if (columnKey === COLUMN_NAMES.USERNAME) {
          return row?.user?.username || '';
        }
        if (columnKey === COLUMN_NAMES.ROLE) {
          return formatRoleLabel(row.role, data.labels);
        }
        return data[columnKey] || '';
      }}
    >
      {(row) => {
        const { user } = row;
        return !readOnly && (
          <TeamMemberRemoveButton
            teamId={row.teamId}
            userId={row.userId}
            disabled={user && user.id === user?.id || row.role === ROLES.TEAM_OWNER}
            onSave={onSave}
          />
        );
      }}
    </SettingsTable>
  );
}

export default TeamMembersTable;