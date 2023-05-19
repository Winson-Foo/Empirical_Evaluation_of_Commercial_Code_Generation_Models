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


// EditButton.js
import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import useMessages from 'hooks/useMessages';

function EditButton({ id, label }) {
  const { formatMessage } = useMessages();

  return (
    <Link href={`/settings/websites/${id}`}>
      <Button>
        <Icon>
          <Icons.Edit />
        </Icon>
        <Text>{formatMessage(label)}</Text>
      </Button>
    </Link>
  );
}

export default EditButton;


// ViewButton.js
import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';

function ViewButton({ id, label }) {
  const { formatMessage } = useMessages();
  const { openExternal } = useConfig();

  return (
    <Link href={`/websites/${id}`} target={openExternal ? '_blank' : null}>
      <Button>
        <Icon>
          <Icons.External />
        </Icon>
        <Text>{formatMessage(label)}</Text>
      </Button>
    </Link>
  );
}

export default ViewButton;

