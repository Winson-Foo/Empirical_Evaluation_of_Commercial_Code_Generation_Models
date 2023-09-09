import React, { useState } from 'react';
import { Loading } from 'react-basics';
import useApi from 'hooks/useApi';
import usePageQuery from 'hooks/usePageQuery';
import useDateRange from 'hooks/useDateRange';
import MetricCard from './MetricCard';
import ErrorMessage from 'components/common/ErrorMessage';
import useMessages from 'hooks/useMessages';
import {
  formatShortTime,
  formatNumber,
  formatLongNumber,
} from 'lib/format';
import styles from './MetricsBar.module.css';

const MetricsBar = ({ websiteId }) => {
  const { formatMessage, labels } = useMessages();
  const { get, useQuery } = useApi();
  const [dateRange] = useDateRange(websiteId);
  const { startDate, endDate, modified } = dateRange;
  const [useLongFormat, setLongFormat] = useState(true);
  const {
    query: { url, referrer, os, browser, device, country, region, city },
  } = usePageQuery();
  const [metrics, setMetrics] = useState({
    pageviews: { value: 0, change: 0 },
    uniques: { value: 0, change: 0 },
    bounces: { value: 0, change: 0 },
    totaltime: { value: 0, change: 0 },
  });

  const fetchWebsiteStats = async () => {
    const response = await get(`/websites/${websiteId}/stats`, {
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

    if (response) {
      setMetrics({
        pageviews: response.pageviews || {},
        uniques: response.uniques || {},
        bounces: response.bounces || {},
        totaltime: response.totaltime || {},
      });
    }
  };

  const { data, error, isLoading, isFetched } = useQuery(
    ['websites:stats', { websiteId, modified, url, referrer, os, browser, device, country, region, city }],
    fetchWebsiteStats
  );

  const formatNumberFunc = useLongFormat ? formatLongNumber : formatNumber;

  const handleSetFormat = () => {
    setLongFormat((useLongFormat) => !useLongFormat);
  }

  const {
    pageviews: { value: pageviewsValue, change: pageviewsChange },
    uniques: { value: uniquesValue, change: uniquesChange },
    bounces: { value: bouncesValue, change: bouncesChange },
    totaltime: { value: totaltimeValue, change: totaltimeChange },
  } = metrics;

  const bounceRateValue =
    uniquesValue ? ((Math.min(uniquesValue, bouncesValue) / uniquesValue) * 100) : 0;

  const bounceRateChange =
    uniquesValue && uniquesChange
      ? ((Math.min(uniquesChange, bouncesChange) / uniquesChange) * 100) -
        ((Math.min(uniquesValue, bouncesValue) / uniquesValue) * 100) || 0
      : 0;

  const averageVisitTimeValue =
    totaltimeValue && pageviewsValue && (pageviewsValue - bouncesValue) > 0
      ? totaltimeValue / (pageviewsValue - bouncesValue)
      : 0;

  const averageVisitTimeChange =
    totaltimeValue && pageviewsValue && (pageviewsValue - bouncesValue) > 0
      ? (totaltimeChange / (pageviewsValue - bouncesValue - pageviewsChange + bouncesChange)) || 0
      : 0;

  const formatTimeFunc = (n) =>
    `${n < 0 ? '-' : ''}${formatShortTime(Math.abs(~~n), ['m', 's'], ' ')}`;

  return (
    <div className={styles.bar} onClick={handleSetFormat}>
      {isLoading && !isFetched && <Loading icon="dots" />}
      {error && <ErrorMessage />}
      {data && isFetched && (
        <>
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.views)}
            value={pageviewsValue}
            change={pageviewsChange}
            format={formatNumberFunc}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.visitors)}
            value={uniquesValue}
            change={uniquesChange}
            format={formatNumberFunc}
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.bounceRate)}
            value={bounceRateValue}
            change={bounceRateChange}
            format={(n) => `${Number(n).toFixed(0)}%`}
            reverseColors
          />
          <MetricCard
            className={styles.card}
            label={formatMessage(labels.averageVisitTime)}
            value={averageVisitTimeValue}
            change={averageVisitTimeChange}
            format={formatTimeFunc}
          />
        </>
      )}
    </div>
  );
};

export default MetricsBar;