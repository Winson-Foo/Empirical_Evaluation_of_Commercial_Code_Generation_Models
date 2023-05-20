import { useState } from 'react';
import { Loading } from 'react-basics';
import ErrorMessage from 'components/common/ErrorMessage';
import useApi from 'hooks/useApi';
import useDateRange from 'hooks/useDateRange';
import usePageQuery from 'hooks/usePageQuery';
import { formatShortTime, formatNumber, formatLongNumber } from 'lib/format';
import MetricCard from './MetricCard';
import useMessages from 'hooks/useMessages';
import styles from './MetricsBar.module.css';

const MetricsBar = ({ websiteId }) => {
  const { formatMessage, labels } = useMessages();
  const { get, useQuery } = useApi();
  const [dateRange] = useDateRange(websiteId);
  const { startDate, endDate, modified } = dateRange;

  const {
    query: { url, referrer, os, browser, device, country, region, city },
  } = usePageQuery();

  const [format, setFormat] = useState(true);

  const { data, error, isLoading, isFetched } = useQuery(
    [
      'websites:stats',
      { websiteId, modified, url, referrer, os, browser, device, country, region, city },
    ],
    () => getWebsiteStats(websiteId, startDate, endDate, url, referrer, os, browser, device, country, region, city, get)
  );

  const formatFunc = format ? formatLongNumber : formatNumber;

  const handleSetFormat = () => setFormat(prevFormat => !prevFormat);

  const { pageviews, uniques, bounces, totaltime } = processData(data);
  const num = Math.min(uniques?.value ?? 0, bounces?.value ?? 0);
  const diffs = {
    pageviews: (pageviews?.value ?? 0) - (pageviews?.change ?? 0),
    uniques: (uniques?.value ?? 0) - (uniques?.change ?? 0),
    bounces: (bounces?.value ?? 0) - (bounces?.change ?? 0),
    totaltime: (totaltime?.value ?? 0) - (totaltime?.change ?? 0),
  };

  return (
    <div className={styles.bar} onClick={handleSetFormat}>
      {isLoading && !isFetched && <Loading icon="dots" />}
      {error && <ErrorMessage />}
      {data && !error && isFetched && (
        <>
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.views)}
            value={pageviews?.value ?? 0}
            change={pageviews?.change ?? 0}
            format={formatFunc}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.visitors)}
            value={uniques?.value ?? 0}
            change={uniques?.change ?? 0}
            format={formatFunc}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.bounceRate)}
            value={uniques?.value ? ((num / uniques.value) * 100).toFixed(0) : 0}
            change={
              uniques?.value && uniques?.change
                ? ((num / uniques.value) * 100 - (Math.min(diffs.uniques, diffs.bounces) / diffs.uniques) * 100 || 0).toFixed(0)
                : 0
            }
            format={n => `${Number(n).toFixed(0)}%`}
            reverseColors
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.averageVisitTime)}
            value={
              totaltime?.value && pageviews?.value
                ? (totaltime.value / (pageviews.value - bounces?.value)) || 0
                : 0
            }
            change={
              totaltime?.value && pageviews?.value
                ? ((diffs.totaltime / (diffs.pageviews - diffs.bounces) - totaltime.value / (pageviews.value - bounces?.value)) * -1 || 0).toFixed(0)
                : 0
            }
            format={n => `${n < 0 ? '-' : ''}${formatShortTime(Math.abs(~~n), ['m', 's'], ' ')}`}
          />
        </>
      )}
    </div>
  );
}

const getWebsiteStats = (websiteId, startDate, endDate, url, referrer, os, browser, device, country, region, city, get) => {
  return get(`/websites/${websiteId}/stats`, {
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
  });
};

const processData = (data) => {
  const defaultValue = { value: 0, change: 0 };
  const {pageviews = defaultValue, uniques = defaultValue, bounces = defaultValue, totaltime = defaultValue} = data || {};
  return {pageviews, uniques, bounces, totaltime};
};

export default MetricsBar;

