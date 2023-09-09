import React from 'react';
import MetricsTable from './MetricsTable';
import useMessages from 'hooks/useMessages';

function EventsTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  const handleDataLoad = React.useCallback(data => {
    props.onDataLoad?.(data);
  }, [props.onDataLoad]);

  return (
    <MetricsTable
      {...props}
      title={formatMessage(labels.events)}
      type="event"
      metric={formatMessage(labels.actions)}
      websiteId={websiteId}
      onDataLoad={handleDataLoad}
    />
  );
}

export default EventsTable;