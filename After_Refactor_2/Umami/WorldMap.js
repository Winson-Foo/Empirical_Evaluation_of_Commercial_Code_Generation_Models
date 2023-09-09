import { useState, useMemo } from 'react';
import classNames from 'classnames';
import colord from 'colord';
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps';

import { ISO_COUNTRIES, MAP_FILE, THEME_COLORS } from 'lib/constants';
import { formatLongNumber } from 'lib/format';
import { percentFilter } from 'lib/filters';
import useCountryNames from 'hooks/useCountryNames';
import useLocale from 'hooks/useLocale';
import useTheme from 'hooks/useTheme';

import HoverTooltip from 'components/common/HoverTooltip';

import styles from './WorldMap.module.css';

function getColorPalette(theme) {
  const { primary, gray100 } = THEME_COLORS[theme];

  return {
    baseColor: primary,
    fillColor: gray100,
    strokeColor: primary,
    hoverColor: primary,
  };
}

function useMapColors(theme) {
  return useMemo(() => getColorPalette(theme), [theme]);
}

function useMapData(data) {
  const metrics = useMemo(() => (data ? percentFilter(data) : []), [data]);

  function getColor(code, colors) {
    if (code === 'AQ') return;

    const country = metrics?.find(({ x }) => x === code);

    if (!country) {
      return colors.fillColor;
    }

    return colord(colors.baseColor)[theme === 'light' ? 'lighten' : 'darken'](
      0.4 * (1.0 - country.z / 100)
    ).toHex();
  }

  function getOpacity(code) {
    return code === 'AQ' ? 0 : 1;
  }

  function handleHover(code) {
    if (code === 'AQ') return;
    const country = metrics?.find(({ x }) => x === code);
    setTooltip(`${countryNames[code]}: ${formatLongNumber(country?.y || 0)} visitors`);
  }

  return {
    handleHover,
    getColor,
    getOpacity,
  };
}

function Map({ geographies, colors, mapData, setTooltip, countryNames, handleHover }) {
  const { locale } = useLocale();

  return (
    <ComposableMap projection="geoMercator">
      <ZoomableGroup zoom={0.8} minZoom={0.7} center={[0, 40]}>
        <Geographies geography={geographies}>
          {({ geographies }) => {
            return geographies.map((geo) => {
              const code = ISO_COUNTRIES[geo.id];

              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={mapData.getColor(code, colors)}
                  stroke={colors.strokeColor}
                  opacity={mapData.getOpacity(code)}
                  style={{
                    default: { outline: 'none' },
                    hover: { outline: 'none', fill: colors.hoverColor },
                    pressed: { outline: 'none' },
                  }}
                  onMouseOver={() => handleHover(code)}
                  onMouseOut={() => setTooltip(null)}
                />
              );
            });
          }}
        </Geographies>
      </ZoomableGroup>
    </ComposableMap>
  );
}

function WorldMap({ data, className }) {
  const [tooltip, setTooltip] = useState(null);
  const { basePath } = useRouter();

  const [theme] = useTheme();
  const colors = useMapColors(theme);

  const { handleHover, getOpacity, getColor } = useMapData(data);

  const countryNames = useCountryNames(locale);

  return (
    <div
      className={classNames(styles.container, className)}
      data-tip=""
      data-for="world-map-tooltip"
    >
      <Map
        geographies={`${basePath}${MAP_FILE}`}
        colors={colors}
        mapData={{ handleHover, getOpacity, getColor }}
        setTooltip={setTooltip}
        countryNames={countryNames}
        handleHover={handleHover}
      />
      {tooltip && <HoverTooltip tooltip={tooltip} />}
    </div>
  );
}

export default WorldMap;

