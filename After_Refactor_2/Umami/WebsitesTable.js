import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import SettingsTable from 'components/common/SettingsTable';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';
import PropTypes from 'prop-types';

const EditButton = ({ id, formatMessage, labels }) => (
  <Link href={`/settings/websites/${id}`}>
    <Button>
      <Icon>
        <Icons.Edit />
      </Icon>
      <Text>{formatMessage(labels.edit)}</Text>
    </Button>
  </Link>
);

const ViewButton = ({ id, formatMessage, labels, target }) => (
  <Link href={`/websites/${id}`} target={target}>
    <Button>
      <Icon>
        <Icons.External />
      </Icon>
      <Text>{formatMessage(labels.view)}</Text>
    </Button>
  </Link>
);

const WebsiteRow = ({ row, openExternal }) => {
  const { id } = row;
  const { formatMessage, labels } = useMessages();
  const target = openExternal ? '_blank' : null;

  return (
    <>
      <EditButton id={id} formatMessage={formatMessage} labels={labels} />
      <ViewButton id={id} formatMessage={formatMessage} labels={labels} target={target} />
    </>
  );
};

WebsiteRow.propTypes = {
  row: PropTypes.object.isRequired,
  openExternal: PropTypes.bool.isRequired,
};

const WebsitesTable = ({ data = [] }) => {
  const { formatMessage, labels } = useMessages();
  const { openExternal } = useConfig();

  const columns = [
    { name: 'name', label: formatMessage(labels.name) },
    { name: 'domain', label: formatMessage(labels.domain) },
    { name: 'action', label: ' ' },
  ];

  return <SettingsTable columns={columns} data={data} rowComponent={WebsiteRow} openExternal={openExternal} />;
};

WebsitesTable.propTypes = {
  data: PropTypes.array.isRequired,
};

export default WebsitesTable;

