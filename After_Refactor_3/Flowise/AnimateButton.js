import PropTypes from 'prop-types';
import { forwardRef } from 'react';
import { motion, useCycle } from 'framer-motion';

const ANIMATION_OFFSET = 10;
const DEFAULT_SCALE = {
  hover: 1,
  tap: 0.9,
};

const ANIMATION_TYPES = {
  ROTATE: 'rotate',
  SLIDE: 'slide',
  SCALE: 'scale',
};

const DIRECTIONS = {
  UP: 'up',
  DOWN: 'down',
  LEFT: 'left',
  RIGHT: 'right',
};

const ANIMATION_VALUES = {
  [ANIMATION_TYPES.SLIDE]: {
    [DIRECTIONS.UP]: { axis: 'y', cycle: useCycle(-ANIMATION_OFFSET, 0) },
    [DIRECTIONS.DOWN]: { axis: 'y', cycle: useCycle(0, ANIMATION_OFFSET) },
    [DIRECTIONS.LEFT]: { axis: 'x', cycle: useCycle(-ANIMATION_OFFSET, 0) },
    [DIRECTIONS.RIGHT]: { axis: 'x', cycle: useCycle(0, ANIMATION_OFFSET) },
  },
  [ANIMATION_TYPES.ROTATE]: { animationProps: { rotate: 360 }, transitionProps: { repeat: Infinity, repeatType: 'loop', duration: 2, repeatDelay: 0 } },
  [ANIMATION_TYPES.SCALE]: {
    whileHover: { scale: DEFAULT_SCALE.hover },
    whileTap: { scale: DEFAULT_SCALE.tap }
  },
};

const AnimateButton = forwardRef(({ children, offset = ANIMATION_OFFSET, type = ANIMATION_TYPES.SCALE, direction = DIRECTIONS.RIGHT, scale = DEFAULT_SCALE }, ref) => {

  const { axis, cycle } = ANIMATION_VALUES[type][direction] ?? {};

  const animateProps = ANIMATION_VALUES[type]?.animationProps;
  const transitionProps = ANIMATION_VALUES[type]?.transitionProps;
  const whileHoverProps = ANIMATION_VALUES[type]?.whileHover;
  const whileTapProps = ANIMATION_VALUES[type]?.whileTap;

  return (
    <motion.div
      ref={ref}
      animate={animateProps}
      transition={transitionProps}
      whileHover={whileHoverProps}
      whileTap={whileTapProps}
      {...(axis && {
        animate: { [axis]: cycle() },
        onHoverEnd: () => cycle(),
        onHoverStart: () => cycle(),
      })}
    >
      {children}
    </motion.div>
  );
});

AnimateButton.propTypes = {
  children: PropTypes.node,
  offset: PropTypes.number,
  type: PropTypes.oneOf(Object.values(ANIMATION_TYPES)),
  direction: PropTypes.oneOf(Object.values(DIRECTIONS)),
  scale: PropTypes.oneOfType([PropTypes.number, PropTypes.object])
};

export default AnimateButton;

