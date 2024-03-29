import { useMemo } from 'react';
import { Loading } from 'react-basics';
import { colord } from 'colord';
import BarChart from './BarChart';
import { getDateArray, getDateLength } from 'lib/date';
import useApi from 'hooks/useApi';
import useDateRange from 'hooks/useDateRange';
import useTimezone from 'hooks/useTimezone';
import usePageQuery from 'hooks/usePageQuery';
import { EVENT_COLORS } from 'lib/constants';

const CHART_HEIGHT = 300;

const getEventDatasets = (eventsData, startDate, endDate, unit) => {
  if (!eventsData) return [];
  const map = eventsData.reduce((obj, { x, t, y }) => {
    if (!obj[x]) {
      obj[x] = [];
    }
    obj[x].push({ x: t, y });
    return obj;
  }, {});
  Object.keys(map).forEach(key => {
    map[key] = getDateArray(map[key], startDate, endDate, unit);
  });
  return Object.keys(map).map((key, index) => {
    const color = colord(EVENT_COLORS[index % EVENT_COLORS.length]);
    return {
      label: key,
      data: map[key],
      lineTension: 0,
      backgroundColor: color.alpha(0.6).toRgbString(),
      borderColor: color.alpha(0.7).toRgbString(),
      borderWidth: 1,
    };
  });
};

const EventsChart = ({ websiteId, className, token }) => {
  const { get, useQuery } = useApi();
  const [{ startDate, endDate, unit, modified }] = useDateRange(websiteId);
  const [timezone] = useTimezone();
  const { query: { url, eventName } } = usePageQuery();

  const { data: eventsData, isLoading } = useQuery(['events', websiteId, modified, eventName], () => get(`/websites/${websiteId}/events`, {
    startAt: +startDate,
    endAt: +endDate,
    unit,
    timezone,
    url,
    eventName,
    token,
  }));

  const datasets = useMemo(() => getEventDatasets(eventsData, startDate, endDate, unit), [eventsData, startDate, endDate, unit]);

  if (isLoading) {
    return <Loading icon="dots" />;
  }

  return (
    <BarChart
      className={className}
      datasets={datasets}
      unit={unit}
      height={CHART_HEIGHT}
      records={getDateLength(startDate, endDate, unit)}
      loading={isLoading}
      stacked
    />
  );
};

export default EventsChart;