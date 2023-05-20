// WebsitesTable.js
import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import SettingsTable from 'components/common/SettingsTable';
import EditButton from 'components/common/EditButton';
import ViewButton from 'components/common/ViewButton';

export function WebsitesTable({ data = [], labels }) {
  const columns = [
    { name: 'name', label: labels.name },
    { name: 'domain', label: labels.domain },
    { name: 'action', label: ' ' },
  ];

  return (
    <SettingsTable columns={columns} data={data}>
      {row => (
        <>
          <EditButton id={row.id} label={labels.edit} />
          <ViewButton id={row.id} label={labels.view} />
        </>
      )}
    </SettingsTable>
  );
}

export default WebsitesTable;