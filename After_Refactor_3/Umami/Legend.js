import { useEffect } from 'react';
import { StatusLight } from 'react-basics';
import { colord } from 'colord';
import classNames from 'classnames';
import useLocale from 'hooks/useLocale';
import useForceUpdate from 'hooks/useForceUpdate';
import styles from './Legend.module.css';

function handleClick(chart, forceUpdate, index) {
  const meta = chart.getDatasetMeta(index);
  meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;
  chart.update();
  forceUpdate();
}

function LegendItem({ text, fillStyle, datasetIndex, hidden, locale }) {
  const color = colord(fillStyle);
  const handleClickItem = () => handleClick(chart, forceUpdate, datasetIndex);

  return (
    <div
      className={classNames(styles.label, { [styles.hidden]: hidden })}
      onClick={handleClickItem}
    >
      <StatusLight color={color.alpha(color.alpha() + 0.2).toHex()}>
        <span className={locale}>{text}</span>
      </StatusLight>
    </div>
  );
}

function Legend({ chart }) {
  const { locale } = useLocale();
  const forceUpdate = useForceUpdate();
  const legendItems = chart?.legend?.legendItems ?? [];

  useEffect(() => {
    forceUpdate();
  }, [locale]);

  if (!legendItems.some(({ text }) => text)) {
    return null;
  }

  return (
    <div className={styles.legend}>
      {legendItems.map(item => (
        <LegendItem key={item.text} {...item} locale={locale} />
      ))}
    </div>
  );
}

export default Legend;

