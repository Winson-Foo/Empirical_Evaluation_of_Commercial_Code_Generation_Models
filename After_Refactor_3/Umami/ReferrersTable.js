import MetricsTable from './MetricsTable';
import FilterLink from 'components/common/FilterLink';
import useMessages from 'hooks/useMessages';

function RenderLink({ referrer, formatMessage, labels }) {
  return (
    <FilterLink
      id="referrer"
      value={referrer}
      externalUrl={`https://${referrer}`}
      label={!referrer && formatMessage(labels.none)}
    />
  );
}

function ReferrersMetricsTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  return (
    <MetricsTable
      {...props}
      title={formatMessage(labels.referrers)}
      type="referrer"
      metric={formatMessage(labels.views)}
      websiteId={websiteId}
      renderLabel={(data) => <RenderLink {...data} formatMessage={formatMessage} labels={labels} />}
    />
  );
}

export default ReferrersMetricsTable;

