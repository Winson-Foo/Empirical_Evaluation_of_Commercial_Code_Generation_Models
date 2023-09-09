import { useState, useMemo } from 'react';
import { useRouter } from 'next/router';
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps';
import classNames from 'classnames';
import { colord } from 'colord';
import HoverTooltip from 'components/common/HoverTooltip';
import { ISO_COUNTRIES, THEME_COLORS, MAP_FILE } from 'lib/constants';
import useTheme from 'hooks/useTheme';
import useCountryNames from 'hooks/useCountryNames';
import useLocale from 'hooks/useLocale';
import { formatLongNumber } from 'lib/format';
import { percentFilter } from 'lib/filters';
import styles from './WorldMap.module.css';

const AQ_CODE = 'AQ'; // Antarctic code
const HIGHLIGHT_PERCENT = 0.4;
const TOOLTIP_ID = 'world-map-tooltip';

function getFillColor(metrics, colors, theme, code) {
  if (code === AQ_CODE) return;
  const country = metrics?.find(({ x }) => x === code);

  if (!country) {
    return colors.fillColor;
  }

  return colord(colors.baseColor)
    [theme === 'light' ? 'lighten' : 'darken'](HIGHLIGHT_PERCENT * (1.0 - country.z / 100))
    .toHex();
}

function getOpacity(code) {
  return code === AQ_CODE ? 0 : 1;
}

function WorldMap({ data, className }) {
  const { basePath } = useRouter();
  const [tooltip, setTooltip] = useState();
  const [theme] = useTheme();
  const colors = useMemo(
    () => ({
      baseColor: THEME_COLORS[theme].primary,
      fillColor: THEME_COLORS[theme].gray100,
      strokeColor: THEME_COLORS[theme].primary,
      hoverColor: THEME_COLORS[theme].primary,
    }),
    [theme],
  );
  const { locale } = useLocale();
  const countryNames = useCountryNames(locale);
  const metrics = useMemo(() => (data ? percentFilter(data) : []), [data]);

  function handleHover(code) {
    if (code === AQ_CODE) return;
    const country = metrics?.find(({ x }) => x === code);
    setTooltip(`${countryNames[code]}: ${formatLongNumber(country?.y || 0)} visitors`);
  }

  function handleMouseOut() {
    setTooltip(null);
  }

  return (
    <div
      className={classNames(styles.container, className)}
      data-tip=""
      data-for={TOOLTIP_ID}
    >
      <ComposableMap projection="geoMercator">
        <ZoomableGroup zoom={0.8} minZoom={0.7} center={[0, 40]}>
          <Geographies geography={`${basePath}${MAP_FILE}`}>
            {({ geographies }) => {
              return geographies.map(({ rsmKey, id, ...geo }) => {
                const code = ISO_COUNTRIES[id];
                const fill = getFillColor(metrics, colors, theme, code);

                return (
                  <Geography
                    key={rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke={colors.strokeColor}
                    opacity={getOpacity(code)}
                    style={{
                      default: { outline: 'none' },
                      hover: { outline: 'none', fill: colors.hoverColor },
                      pressed: { outline: 'none' },
                    }}
                    onMouseOver={() => handleHover(code)}
                    onMouseOut={handleMouseOut}
                  />
                );
              });
            }}
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>
      {tooltip && <HoverTooltip tooltip={tooltip} />}
    </div>
  );
}

export default WorldMap;

