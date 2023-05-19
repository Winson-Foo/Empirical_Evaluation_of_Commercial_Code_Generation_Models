import MetricsTable from './MetricsTable';
import useMessages from 'hooks/useMessages';

const SCREEN_TITLE = 'screens';
const VISITORS_METRIC = 'visitors';

export function ScreenTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();
  const screenTitle = formatMessage(labels[SCREEN_TITLE]);
  const metric = formatMessage(labels[VISITORS_METRIC]);

  return (
    <MetricsTable
      {...props}
      title={screenTitle}
      type={SCREEN_TITLE}
      metric={metric}
      websiteId={websiteId}
    />
  );
}

export default ScreenTable;

