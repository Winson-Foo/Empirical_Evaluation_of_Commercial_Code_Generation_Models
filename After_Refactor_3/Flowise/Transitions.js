import PropTypes from 'prop-types'
import { forwardRef } from 'react'
import { Collapse, Fade, Box, Grow, Slide, Zoom } from '@mui/material'

export const POSITION_OPTIONS = {
    topLeft: 'top-left',
    topRight: 'top-right',
    top: 'top',
    bottomLeft: 'bottom-left',
    bottomRight: 'bottom-right',
    bottom: 'bottom'
}

export const TYPE_OPTIONS = {
    grow: 'grow',
    fade: 'fade',
    collapse: 'collapse',
    slide: 'slide',
    zoom: 'zoom'
}

const getPositionSX = (position) => {
    switch (position) {
        case POSITION_OPTIONS.topRight:
            return { transformOrigin: 'top right' }
        case POSITION_OPTIONS.top:
            return { transformOrigin: 'top' }
        case POSITION_OPTIONS.bottomLeft:
            return { transformOrigin: 'bottom left' }
        case POSITION_OPTIONS.bottomRight:
            return { transformOrigin: 'bottom right' }
        case POSITION_OPTIONS.bottom:
            return { transformOrigin: 'bottom' }
        case POSITION_OPTIONS.topLeft:
        default:
            return { transformOrigin: '0 0 0' }
    }
}

const Transitions = forwardRef(({ children, type = TYPE_OPTIONS.grow, position = POSITION_OPTIONS.topLeft, direction = 'up', ...others }, ref) => {
    const positionSX = getPositionSX(position)

    switch (type) {
        case TYPE_OPTIONS.grow:
            return (
                <Box ref={ref}>
                    <Grow {...others}>
                        <Box sx={positionSX}>{children}</Box>
                    </Grow>
                </Box>
            )
        case TYPE_OPTIONS.collapse:
            return (
                <Box ref={ref}>
                    <Collapse {...others} sx={positionSX}>
                        {children}
                    </Collapse>
                </Box>
            )
        case TYPE_OPTIONS.fade:
            return (
                <Box ref={ref}>
                    <Fade
                        {...others}
                        timeout={{
                            appear: 500,
                            enter: 600,
                            exit: 400
                        }}
                    >
                        <Box sx={positionSX}>{children}</Box>
                    </Fade>
                </Box>
            )
        case TYPE_OPTIONS.slide:
            return (
                <Box ref={ref}>
                    <Slide
                        {...others}
                        timeout={{
                            appear: 0,
                            enter: 400,
                            exit: 200
                        }}
                        direction={direction}
                    >
                        <Box sx={positionSX}>{children}</Box>
                    </Slide>
                </Box>
            )
        case TYPE_OPTIONS.zoom:
            return (
                <Box ref={ref}>
                    <Zoom {...others}>
                        <Box sx={positionSX}>{children}</Box>
                    </Zoom>
                </Box>
            )
        default:
            return null
    }
})

Transitions.propTypes = {
    children: PropTypes.node,
    type: PropTypes.oneOf(Object.values(TYPE_OPTIONS)),
    position: PropTypes.oneOf(Object.values(POSITION_OPTIONS)),
    direction: PropTypes.oneOf(['up', 'down', 'left', 'right'])
}

export default Transitions

