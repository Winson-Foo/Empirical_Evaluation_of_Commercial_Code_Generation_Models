import { useEffect } from 'react';
import { StatusLight } from 'react-basics';
import { colord } from 'colord';
import classNames from 'classnames';
import useLocale from 'hooks/useLocale';
import useForceUpdate from 'hooks/useForceUpdate';
import styles from './Legend.module.css';

function handleClick(chart, index, forceUpdate) {
  const datasetMeta = chart.getDatasetMeta(index);

  // Toggle the hidden property of the datasetMeta object
  datasetMeta.hidden = datasetMeta.hidden === null ? !chart.data.datasets[index].hidden : null;

  chart.update();

  forceUpdate();
}

function useLocaleUpdateHook(locale, forceUpdate) {
  useEffect(() => {
    forceUpdate(); // Trigger a re-render when locale changes
  }, [locale, forceUpdate]);
}

function Legend({ chart }) {
  const { locale } = useLocale();
  const forceUpdate = useForceUpdate();

  const chartData = chart?.data?.datasets;
  const legendItems = chart?.legend?.legendItems;

  // Return null if there is no legend item with text
  if (!legendItems?.find(({ text }) => text)) {
    return null;
  }

  useLocaleUpdateHook(locale, forceUpdate);

  return (
    <div className={styles.legend}>
      {legendItems.map(({ text, fillStyle, datasetIndex, hidden }) => {
        const color = colord(fillStyle);

        return (
          <div
            key={text}
            className={classNames(styles.label, { [styles.hidden]: hidden })}
            onClick={() => handleClick(chart, datasetIndex, forceUpdate)}
          >
            <StatusLight color={color.alpha(color.alpha() + 0.2).toHex()}>
              <span className={locale}>{text}</span>
            </StatusLight>
          </div>
        );
      })}
    </div>
  );
}

export default Legend;

