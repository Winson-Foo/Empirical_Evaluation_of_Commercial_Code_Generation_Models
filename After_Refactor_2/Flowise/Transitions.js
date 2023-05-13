import PropTypes from 'prop-types';
import { forwardRef } from 'react';
import { Collapse, Fade, Box, Grow, Slide, Zoom } from '@mui/material';

const positions = {
  'top-left': '0 0 0',
  'top-right': 'top right',
  'top': 'top',
  'bottom-left': 'bottom left',
  'bottom-right': 'bottom right',
  'bottom': 'bottom'
};

const timeouts = {
  appear: 500,
  enter: 600,
  exit: 400
};

const slideTimeouts = {
  appear: 0,
  enter: 400,
  exit: 200
};

const Transitions = forwardRef(({ children, type = 'grow', position = 'top-left', direction = 'up', ...others }, ref) => {
  const positionSX = { transformOrigin: positions[position] };

  const transitions = {
    grow: <Grow {...others}><Box sx={positionSX}>{children}</Box></Grow>,
    collapse: <Collapse {...others} sx={positionSX}>{children}</Collapse>,
    fade: <Fade {...others} timeout={timeouts}><Box sx={positionSX}>{children}</Box></Fade>,
    slide: <Slide {...others} timeout={slideTimeouts} direction={direction}><Box sx={positionSX}>{children}</Box></Slide>,
    zoom: <Zoom {...others}><Box sx={positionSX}>{children}</Box></Zoom>
  }

  return <Box ref={ref}>{transitions[type]}</Box>;
});

Transitions.propTypes = {
  children: PropTypes.node,
  type: PropTypes.oneOf(['grow', 'fade', 'collapse', 'slide', 'zoom']),
  position: PropTypes.oneOf(['top-left', 'top-right', 'top', 'bottom-left', 'bottom-right', 'bottom']),
  direction: PropTypes.oneOf(['up', 'down', 'left', 'right'])
};

export default Transitions;

