import { useState } from 'react';
import { Loading } from 'react-basics';
import useApi from 'hooks/useApi';
import useDateRange from 'hooks/useDateRange';
import usePageQuery from 'hooks/usePageQuery';
import { formatShortTime, formatNumber, formatLongNumber } from 'lib/format';
import useMessages from 'hooks/useMessages';
import ErrorMessage from 'components/common/ErrorMessage';
import MetricCard from './MetricCard';
import styles from './MetricsBar.module.css';

const DEFAULT_FORMAT = true;
const BOUNCE_RATE_LABEL = 'Bounce Rate';
const AVERAGE_VISIT_TIME_LABEL = 'Average Visit Time';

function MetricsBar({ websiteId }) {
  // Hooks
  const { formatMessage, labels } = useMessages();
  const { get, useQuery } = useApi();
  const [dateRange] = useDateRange(websiteId);
  const { startDate, endDate, modified } = dateRange;
  const [useFormat, setUseFormat] = useState(DEFAULT_FORMAT);
  const {
    query: { url, referrer, os, browser, device, country, region, city },
  } = usePageQuery();

  // Query data
  const { data, error, isLoading, isFetched } = useQuery(
    [
      'websites:stats',
      { websiteId, modified, url, referrer, os, browser, device, country, region, city },
    ],
    () =>
      get(`/websites/${websiteId}/stats`, {
        startAt: +startDate,
        endAt: +endDate,
        url,
        referrer,
        os,
        browser,
        device,
        country,
        region,
        city,
      }),
  );

  // Format functions
  const formatNum = useFormat ? formatLongNumber : formatNumber;
  const formatPercent = n => Number(n).toFixed(0) + '%';
  const formatTime = n =>
    `${n < 0 ? '-' : ''}${formatShortTime(Math.abs(~~n), ['m', 's'], ' ')}`;

  // Helper functions
  const getDataValue = (data, key) => data?.[key]?.value ?? 0;
  const getDataChange = (data, key) =>
    (data?.[key]?.value ?? 0) - (data?.[key]?.change ?? 0);
  const getBounceRate = (data) => {
    const uniques = getDataValue(data, 'uniques');
    const bounces = Math.min(getDataValue(data, 'bounces'), uniques);
    return uniques ? (bounces / uniques) * 100 : 0;
  };
  const getDiff = (data, key) => ({
    [key]: getDataValue(data, key) - getDataChange(data, key),
  });

  // Event handlers
  function handleSetUseFormat() {
    setUseFormat(useFormat => !useFormat);
  }

  // Data variables
  const pageviewsData = { value: getDataValue(data, 'pageviews'), change: getDataChange(data, 'pageviews') };
  const uniquesData = { value: getDataValue(data, 'uniques'), change: getDataChange(data, 'uniques') };
  const bouncesData = { value: getDataValue(data, 'bounces'), change: getDataChange(data, 'bounces') };
  const totaltimeData = { value: getDataValue(data, 'totaltime'), change: getDataChange(data, 'totaltime') };
  const bounceRateValue = getBounceRate(data);
  const bounceRateDiff =
    uniquesData.value && uniquesData.change
      ? (bounceRateValue - getBounceRate(getDiff(data, 'uniques', 'bounces'))) || 0
      : 0;
  const avgTimeValue = totaltimeData.value / (pageviewsData.value - bouncesData.value);
  const avgTimeDiff =
    totaltimeData.value && pageviewsData.value
      ? (getDiff(totaltimeData - bouncesData.value - diff.totaltime - diff.bounces) / 
         (getDiff(pageviewsData - bouncesData.value - diff.pageviews) - getDiff(totaltimeData - bouncesData.value - diff.bounces))) * -1 || 0
      : 0;

  // Render
  return (
    <div className={styles.bar} onClick={handleSetUseFormat}>
      {isLoading && !isFetched && <Loading icon="dots" />}
      {error && <ErrorMessage />}
      {data && !error && isFetched && (
        <>
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.views)}
            value={pageviewsData.value}
            change={pageviewsData.change}
            format={formatNum}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.visitors)}
            value={uniquesData.value}
            change={uniquesData.change}
            format={formatNum}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.bounceRate)}
            value={bounceRateValue}
            change={bounceRateDiff}
            format={formatPercent}
            reverseColors
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.averageVisitTime)}
            value={avgTimeValue}
            change={avgTimeDiff}
            format={formatTime}
          />
        </>
      )}
    </div>
  );
}

export default MetricsBar;

