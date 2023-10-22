import React, { useState } from 'react';
import PropTypes from 'prop-types';
import Tooltip from 'react-basics/Tooltip';
import styles from './HoverTooltip.module.css';

const useTooltipPosition = () => {
  const [position, setPosition] = useState({ x: -1000, y: -1000 });

  const handleMouseMove = (e) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  React.useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return position;
};

const HoverTooltip = ({ label }) => {
  const position = useTooltipPosition();

  return (
    <div className={styles.tooltip} style={{ left: position.x, top: position.y }}>
      <Tooltip position="top" action="none" label={label} />
    </div>
  );
};

HoverTooltip.propTypes = {
  label: PropTypes.string.isRequired,
};

export default HoverTooltip;