import MetricsTable from './MetricsTable';
import useMessages from 'hooks/useMessages';

function getTableTitle(labels) {
  return labels.screens;
}

function getMetricLabel(labels) {
  return labels.visitors;
}

export function ScreenTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();
  const title = getTableTitle(labels);
  const metric = getMetricLabel(labels);

  return (
    <MetricsTable
      {...props}
      title={formatMessage(title)}
      type="screen"
      metric={formatMessage(metric)}
      websiteId={websiteId}
    />
  );
}

export default ScreenTable;

