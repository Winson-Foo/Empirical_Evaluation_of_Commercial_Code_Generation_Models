import MetricsTable from './MetricsTable';
import useMessages from 'hooks/useMessages';

const EventsTable = ({ websiteId, ...props }) => {
  const { formatMessage, labels } = useMessages();

  const handleDataLoad = (data) => {
    props.onDataLoad?.(data);
  }

  const { type, onDataLoad, ...rest } = props;

  return (
    <MetricsTable
      {...rest}
      title={formatMessage(labels.events)}
      type="event"
      metric={formatMessage(labels.actions)}
      websiteId={websiteId}
      onDataLoad={handleDataLoad}
    />
  );
}

export default EventsTable;