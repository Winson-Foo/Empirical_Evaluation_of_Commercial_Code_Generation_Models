import React from 'react';
import PropTypes from 'prop-types';
import { useMemo } from 'react';
import { colord } from 'colord';
import BarChart from './BarChart';
import { THEME_COLORS } from 'lib/constants';
import useTheme from 'hooks/useTheme';
import useMessages from 'hooks/useMessages';
import useLocale from 'hooks/useLocale';

function getPageviewsChartColors(theme) {
  const primaryColor = colord(THEME_COLORS[theme].primary);
  return {
    views: {
      hoverBackgroundColor: primaryColor.alpha(0.7).toRgbString(),
      backgroundColor: primaryColor.alpha(0.4).toRgbString(),
      borderColor: primaryColor.alpha(0.7).toRgbString(),
      hoverBorderColor: primaryColor.toRgbString(),
    },
    visitors: {
      hoverBackgroundColor: primaryColor.alpha(0.9).toRgbString(),
      backgroundColor: primaryColor.alpha(0.6).toRgbString(),
      borderColor: primaryColor.alpha(0.9).toRgbString(),
      hoverBorderColor: primaryColor.toRgbString(),
    },
  };
}

function getDatasets(data, formatMessage, labels, colors) {
  if (!data) return [];

  return [
    {
      label: formatMessage(labels.uniqueVisitors),
      data: data.sessions,
      borderWidth: 1,
      ...colors.visitors,
    },
    {
      label: formatMessage(labels.pageViews),
      data: data.pageviews,
      borderWidth: 1,
      ...colors.views,
    },
  ];
}

function PageviewsChart(props) {
  const { websiteId, data, unit, records, className, loading } = props;
  const { formatMessage, labels } = useMessages();
  const [theme] = useTheme();
  const { locale } = useLocale();

  const colors = useMemo(() => getPageviewsChartColors(theme), [theme]);

  const datasets = useMemo(() => getDatasets(data, formatMessage, labels, colors), [data, formatMessage, labels, colors]);

  return (
    <BarChart
      key={websiteId}
      className={className}
      datasets={datasets}
      unit={unit}
      records={records}
      loading={loading}
    />
  );
}

PageviewsChart.propTypes = {
  websiteId: PropTypes.string.isRequired,
  data: PropTypes.object,
  unit: PropTypes.string,
  records: PropTypes.number,
  className: PropTypes.string,
  loading: PropTypes.bool,
};

export default PageviewsChart;

