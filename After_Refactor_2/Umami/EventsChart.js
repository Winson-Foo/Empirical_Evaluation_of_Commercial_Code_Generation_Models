// constants
const EVENT_COLORS = [...]; // define colors for different events here

// components
function EventsChart({ websiteId, className, token }) {
  const { get, useQuery } = useApi();
  const [{ startDate, endDate, unit, modified }] = useDateRange(websiteId);
  const [timezone] = useTimezone();
  const {
    query: { url, eventName },
  } = usePageQuery();

  const { data, isLoading } = useQuery(['events', websiteId, modified, eventName], () =>
    getEventsData(websiteId, startDate, endDate, unit, timezone, url, eventName, token),
  );

  const datasets = generateDatasets(data, isLoading, startDate, endDate, unit);

  return (
    <BarChart
      className={className}
      datasets={datasets}
      unit={unit}
      height={300}
      records={getDateLength(startDate, endDate, unit)}
      loading={isLoading}
      stacked
    />
  );
}

function getEventsData(websiteId, startDate, endDate, unit, timezone, url, eventName, token) {
  return get(`/websites/${websiteId}/events`, {
    startAt: +startDate,
    endAt: +endDate,
    unit,
    timezone,
    url,
    eventName,
    token,
  });
}

function generateDatasets(data, isLoading, startDate, endDate, unit) {
  if (!data) return [];
  if (isLoading) return data;

  const eventsData = processData(data, startDate, endDate, unit);

  return eventsData.map((event, index) => {
    const color = colord(EVENT_COLORS[index % EVENT_COLORS.length]);

    return {
      label: event.name,
      data: event.values,
      backgroundColor: color.alpha(0.6).toRgbString(),
      borderColor: color.alpha(0.7).toRgbString(),
      borderWidth: 1,
    };
  });
}

function processData(data, startDate, endDate, unit) {
  const eventsMap = data.reduce((map, event) => {
    if (!map[event.name]) {
      map[event.name] = [];
    }

    map[event.name].push(event);

    return map;
  }, {});

  const eventsData = Object.keys(eventsMap).map(name => {
    const events = eventsMap[name];
    const values = getDateArray(events, startDate, endDate, unit);
    
    return { name, values };
  });

  return eventsData;
}

export default EventsChart;