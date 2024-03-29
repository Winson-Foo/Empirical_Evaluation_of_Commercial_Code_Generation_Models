// TeamMembersTable.js

import React from 'react';
import SettingsTable from 'components/common/SettingsTable';
import TeamMemberRemoveButton from './TeamMemberRemoveButton';
import * as Constants from 'constants';

import useMessages from 'hooks/useMessages';
import useUser from 'hooks/useUser';

import { UserRole } from 'types';

import { TableColumn } from 'components/common/Table';

type TeamMembersTableProps = {
  data: UserRole[];
  onSave: () => void;
  readOnly: boolean;
};

function TeamMembersTable({ data = [], onSave, readOnly }: TeamMembersTableProps) {
  const { formatMessage, labels } = useMessages();
  const { user } = useUser();

  const columns: TableColumn[] = [
    { name: 'username', label: formatMessage(labels.username) },
    { name: 'role', label: formatMessage(labels.role) },
    { name: 'action', label: ' ' },
  ];

  const renderCell = (row: any, data: any, key: string) => {
    if (key === 'username') {
      return row?.user?.username;
    }
    if (key === 'role') {
      return formatMessage(
        labels[
          Object.keys(Constants.UserRole).find(key => Constants.UserRole[key] === row.role) ||
            labels.unknown
        ],
      );
    }
    return data[key];
  };

  const renderRowAction = (row: UserRole) => {
    if (!readOnly) {
      return (
        <TeamMemberRemoveButton
          teamId={row.teamId}
          userId={row.userId}
          disabled={user.id === row?.user?.id || row.role === Constants.UserRole.TeamOwner}
          onSave={onSave}
        />
      );
    }
  };

  return (
    <SettingsTable data={data} columns={columns} cellRender={renderCell}>
      {renderRowAction}
    </SettingsTable>
  );
}

export default TeamMembersTable;