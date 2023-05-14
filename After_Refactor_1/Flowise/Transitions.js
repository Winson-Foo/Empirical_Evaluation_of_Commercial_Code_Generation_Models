import PropTypes from 'prop-types'
import { forwardRef } from 'react'

// material-ui
import { Collapse, Fade, Box, Grow, Slide, Zoom } from '@mui/material'

// Extract switch statement to separate function
const getPositionSX = (position) => {
  const transformOrigins = {
    'top-right': 'top right',
    top: 'top',
    'bottom-left': 'bottom left',
    'bottom-right': 'bottom right',
    bottom: 'bottom',
    'top-left': '0 0 0',
  }
  return { transformOrigin: transformOrigins[position] }
}

const Transitions = forwardRef(
  // Use object destructuring to simplify code
  ({ children, position = 'top-left', type = 'grow', direction = 'up' }, ref) => {
    const transformOrigin = getPositionSX(position)

    // Simplify code by removing unnecessary spread operators
    return (
      <Box ref={ref}>
        {type === 'grow' && (
          <Grow sx={transformOrigin} {...props}>
            {children}
          </Grow>
        )}
        {type === 'collapse' && (
          <Collapse sx={transformOrigin} {...props}>
            {children}
          </Collapse>
        )}
        {type === 'fade' && (
          <Fade
            sx={transformOrigin}
            timeout={{ appear: 500, enter: 600, exit: 400 }}
            {...props}
          >
            {children}
          </Fade>
        )}
        {type === 'slide' && (
          <Slide
            sx={transformOrigin}
            timeout={{ appear: 0, enter: 400, exit: 200 }}
            direction={direction}
            {...props}
          >
            {children}
          </Slide>
        )}
        {type === 'zoom' && (
          <Zoom sx={transformOrigin} {...props}>
            {children}
          </Zoom>
        )}
      </Box>
    )
  }
)

Transitions.propTypes = {
  children: PropTypes.node,
  type: PropTypes.oneOf(['grow', 'fade', 'collapse', 'slide', 'zoom']),
  position: PropTypes.oneOf([
    'top-left',
    'top-right',
    'top',
    'bottom-left',
    'bottom-right',
    'bottom',
  ]),
  direction: PropTypes.oneOf(['up', 'down', 'left', 'right']),
}

export default Transitions

