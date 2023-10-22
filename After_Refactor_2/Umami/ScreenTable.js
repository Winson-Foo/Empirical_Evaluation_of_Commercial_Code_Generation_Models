import { useMessages } from 'hooks/useMessages';
import MetricsTable from './MetricsTable';

type ScreenTableProps = {
  websiteId: string;
}

function ScreenTable({ websiteId, ...otherProps }: ScreenTableProps) {
  const { formatMessage, labels } = useMessages();
  const title = formatMessage(labels.screens);
  const type = "screen";
  const metric = formatMessage(labels.visitors);

  return (
    <MetricsTable
      {...otherProps}
      title={title}
      type={type}
      metric={metric}
      websiteId={websiteId}
    />
  );
}

export default ScreenTable;

