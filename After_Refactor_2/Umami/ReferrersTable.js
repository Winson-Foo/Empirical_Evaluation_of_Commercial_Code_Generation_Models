import { useMemo } from 'react';
import { MetricsTable } from './MetricsTable';
import { FilterLink } from 'components/common/FilterLink';
import { useMessages } from 'hooks/useMessages';

const LABELS = {
  referrers: 'Referrers',
  views: 'Views',
  none: 'None',
};

const ReferrersTable = ({ websiteId }) => {
  const { formatMessage } = useMessages();
  
  const renderLabel = useMemo(() => {
    return ({ x: referrer }) => {
      return (
        <FilterLink
          id="referrer"
          value={referrer}
          externalUrl={`https://${referrer}`}
          label={!referrer && formatMessage(LABELS.none)}
        />
      );
    };
  }, [formatMessage]);

  return (
    <MetricsTable
      title={formatMessage(LABELS.referrers)}
      type="referrer"
      metric={formatMessage(LABELS.views)}
      websiteId={websiteId}
      renderLabel={renderLabel}
    />
  );
};

export default ReferrersTable;

