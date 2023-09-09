import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import SettingsTable from 'components/common/SettingsTable';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';

const EDIT_LABEL = 'edit';
const VIEW_LABEL = 'view';

function EditButton({ label, id }) {
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

function ViewButton({ label, id, target }) {
  const { formatMessage } = useMessages();
  const { openExternal } = useConfig();

  return (
    <Link href={`/websites/${id}`} target={target || (openExternal ? '_blank' : null)}>
      <Button>
        <Icon>
          <Icons.External />
        </Icon>
        <Text>{formatMessage(label)}</Text>
      </Button>
    </Link>
  );
}

function getTableColumns() {
  const { formatMessage, labels } = useMessages();

  return [
    { name: 'name', label: formatMessage(labels.name) },
    { name: 'domain', label: formatMessage(labels.domain) },
    { name: 'action', label: ' ' },
  ];
}

function getTableRowButtons(row) {
  const { id } = row;

  return (
    <>
      <EditButton label={EDIT_LABEL} id={id} />
      <ViewButton label={VIEW_LABEL} id={id} />
    </>
  );
}

export function WebsitesTable({ data = [] }) {
  const columns = getTableColumns();

  return (
    <SettingsTable columns={columns} data={data}>
      {getTableRowButtons}
    </SettingsTable>
  );
}

export default WebsitesTable;