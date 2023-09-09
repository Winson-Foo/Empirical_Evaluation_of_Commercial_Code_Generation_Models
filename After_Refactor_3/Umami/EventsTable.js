import MetricsTable from './MetricsTable';
import useMessages from 'hooks/useMessages';

export function EventsTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  function handleDataLoad(data) {
    props.onDataLoad?.(data);
  }

  function getTitle() {
    return formatMessage(labels.events);
  }

  function getMetric() {
    return formatMessage(labels.actions);
  }

  return (
    <MetricsTable
      {...props}
      title={getTitle()}
      type="event"
      metric={getMetric()}
      websiteId={websiteId}
      onDataLoad={handleDataLoad}
    />
  );
}

export default EventsTable; 