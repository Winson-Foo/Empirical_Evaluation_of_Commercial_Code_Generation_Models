import { useEffect, useState } from 'react';
import { Tooltip } from 'react-basics';
import styles from './HoverTooltip.module.css';

function useMouseMove() {
  const [position, setPosition] = useState({ x: -1000, y: -1000 });

  useEffect(() => {
    const handler = e => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    document.addEventListener('mousemove', handler);

    return () => {
      document.removeEventListener('mousemove', handler);
    };
  }, []);

  return position;
}

function HoverTooltip({ tooltip }) {
  const position = useMouseMove();

  return (
    <div className={styles.tooltip} style={{ left: position.x, top: position.y }}>
      <Tooltip position="top" action="none" label={tooltip} />
    </div>
  );
}

export default HoverTooltip;

