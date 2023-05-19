import { useEffect, useState } from 'react';
import { Tooltip } from 'react-basics';
import styles from './HoverTooltip.module.css';

const HOVER_POSITION = { x: -1000, y: -1000 };

export function HoverTooltip({ tooltip }) {
  const [position, setPosition] = useState(HOVER_POSITION);

  useEffect(() => {
    const handleMouseMove = (event) => {
      const { clientX: x, clientY: y } = event;
      setPosition({ x, y });
    };
    document.addEventListener('mousemove', handleMouseMove);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const tooltipStyle = { left: position.x, top: position.y };

  return (
    <div className={styles.tooltip} style={tooltipStyle}>
      <Tooltip label={tooltip} action="none" position="top" />
    </div>
  );
}

export default HoverTooltip;

