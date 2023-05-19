import { useEffect } from 'react';
import { StatusLight } from 'react-basics';
import { colord } from 'colord';
import classNames from 'classnames';
import useLocale from 'hooks/useLocale';
import useForceUpdate from 'hooks/useForceUpdate';
import styles from './Legend.module.css';

function renderLegendItem({ text, fillStyle, datasetIndex, hidden }, handleClick, locale) {
  const color = colord(fillStyle);

  return (
    <div
      key={text}
      className={classNames(styles.label, { [styles.hidden]: hidden })}
      onClick={() => handleClick(datasetIndex)}
    >
      <StatusLight color={color.alpha(color.alpha() + 0.2).toHex()}>
        <span className={locale}>{text}</span>
      </StatusLight>
    </div>
  );
}

export function Legend({ chart }) {
  const { locale } = useLocale();
  const forceUpdate = useForceUpdate();

  const handleClick = index => {
    const meta = chart.getDatasetMeta(index);

    meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;

    chart.update();

    forceUpdate();
  };

  useEffect(() => {
    const chartUpdate = () => {
      chart.update();
      forceUpdate();
    };

    chart.on('update', chartUpdate);

    return () => chart.off('update', chartUpdate);
  }, [chart]);

  const { legend } = chart;

  if (!legend?.legendItems.find(({ text }) => text)) {
    return null;
  }

  return (
    <div className={styles.legend}>
      {legend.legendItems.map(item => renderLegendItem(item, handleClick, locale))}
    </div>
  );
}

export default Legend;

