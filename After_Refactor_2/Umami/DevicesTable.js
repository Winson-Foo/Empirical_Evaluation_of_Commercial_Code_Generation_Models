import MetricsTable from './MetricsTable';
import FilterLink from 'components/common/FilterLink';
import useMessages from 'hooks/useMessages';
import { getDeviceLinkLabel } from 'utils/linkLabels';

function DevicesTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();
  const metric = formatMessage(labels.visitors);
  const type = 'device';
  const title = formatMessage(labels.devices);

  function renderLabel({ x: device }) {
    const label = getDeviceLinkLabel({ formatMessage, labels, device });
    return <FilterLink id="device" value={device} label={label} />;
  }

  return (
    <MetricsTable
      {...props}
      title={title}
      type={type}
      metric={metric}
      websiteId={websiteId}
      renderLabel={renderLabel}
    />
  );
}

DevicesTable.defaultProps = {
  websiteId: 'default-website-id',
};

export default DevicesTable;

export function getDeviceLinkLabel({ formatMessage, labels, device }) {
  const label = labels[device] || labels.unknown;
  return formatMessage(`device.${label}`);
} 