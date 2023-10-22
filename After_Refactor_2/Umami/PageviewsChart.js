import { useMemo } from 'react';
import { colord } from 'colord';
import BarChart from './BarChart';
import useTheme from 'hooks/useTheme';
import useMessages from 'hooks/useMessages';
import useLocale from 'hooks/useLocale';
import { THEME_COLORS } from 'lib/constants';

const PRIMARY_COLOR_ALPHA = {
  views: {
    hover: 0.7,
    background: 0.4,
    border: 0.7,
  },
  visitors: {
    hover: 0.9,
    background: 0.6,
    border: 0.9,
  },
};

function PageviewsChart({
  websiteId,
  data,
  unit,
  records,
  className,
  loading,
  ...props
}) {
  const { formatMessage, labels } = useMessages();
  const [theme] = useTheme();
  const { locale } = useLocale();

  const primaryColor = useMemo(() => {
    return colord(THEME_COLORS[theme].primary);
  }, [theme]);

  const colors = useMemo(() => {
    return {
      views: {
        hoverBackgroundColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.views.hover).toRgbString(),
        backgroundColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.views.background).toRgbString(),
        borderColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.views.border).toRgbString(),
        hoverBorderColor: primaryColor.toRgbString(),
      },
      visitors: {
        hoverBackgroundColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.visitors.hover).toRgbString(),
        backgroundColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.visitors.background).toRgbString(),
        borderColor: primaryColor.alpha(PRIMARY_COLOR_ALPHA.visitors.border).toRgbString(),
        hoverBorderColor: primaryColor.toRgbString(),
      },
    };
  }, [theme, primaryColor]);

  const datasets = useMemo(() => {
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
  }, [data, locale, colors, formatMessage, labels]);

  return (
    <BarChart
      {...props}
      key={websiteId}
      className={className}
      datasets={datasets}
      unit={unit}
      records={records}
      loading={loading}
    />
  );
}

export default PageviewsChart;

