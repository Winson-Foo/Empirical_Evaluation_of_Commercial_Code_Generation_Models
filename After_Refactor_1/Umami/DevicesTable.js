import MetricsTable from './MetricsTable';
import FilterLink from 'components/common/FilterLink';
import useMessages from 'hooks/useMessages';

const DEVICES_LABEL_ID = 'devices';
const DEVICE_FILTER_ID = 'device';
const VISITORS_LABEL_ID = 'visitors';
const UNKNOWN_LABEL_ID = 'unknown';

function DeviceLink({ formatMessage, device }) {
  const deviceLabel = formatMessage(labels[device] || labels[UNKNOWN_LABEL_ID]);
  const deviceValue = labels[device] && device;

  return (
    <FilterLink id={DEVICE_FILTER_ID} value={deviceValue} label={deviceLabel} />
  );
}

function DevicesTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  const devicesLabel = formatMessage(labels[DEVICES_LABEL_ID]);
  const visitorsMetric = formatMessage(labels[VISITORS_LABEL_ID]);

  return (
    <MetricsTable
      {...props}
      title={devicesLabel}
      type={DEVICE_FILTER_ID}
      metric={visitorsMetric}
      websiteId={websiteId}
      renderLabel={(device) => (
        <DeviceLink formatMessage={formatMessage} labels={labels} device={device} />
      )}
    />
  );
}

export default DevicesTable;