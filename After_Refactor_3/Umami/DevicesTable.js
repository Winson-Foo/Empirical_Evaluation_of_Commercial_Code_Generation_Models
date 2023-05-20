import React from 'react';
import PropTypes from 'prop-types';
import MetricsTable from './MetricsTable';
import FilterLink from 'components/common/FilterLink';
import useMessages from 'hooks/useMessages';

function renderDeviceLink({ x: device, labels, formatMessage }) {
  const label = labels[device] || labels.unknown;
  const value = labels[device] && device;

  return (
    <FilterLink
      id="device"
      value={value}
      label={formatMessage(label)}
    />
  );
}

function DevicesMetricTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  return (
    <MetricsTable
      {...props}
      title={formatMessage(labels.devices)}
      type="device"
      metric={formatMessage(labels.visitors)}
      websiteId={websiteId}
      renderLabel={data => renderDeviceLink({ ...data, labels, formatMessage })}
    />
  );
}

DevicesMetricTable.propTypes = {
  websiteId: PropTypes.string.isRequired,
};

export default DevicesMetricTable;

