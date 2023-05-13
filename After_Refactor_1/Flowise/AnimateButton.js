import PropTypes from 'prop-types'
import { forwardRef } from 'react'
// third-party
import { motion, useCycle } from 'framer-motion'

// ==============================|| ANIMATION BUTTON ||============================== //

const AnimateButton = forwardRef(function AnimateButton({ children, type = 'scale', direction = 'right', offset = 10, scale = { hover: 1, tap: 0.9 } }, ref) {
    const directionValues = {
        up: { translateY: -offset },
        down: { translateY: offset },
        left: { translateX: -offset },
        right: { translateX: offset }
    }

    const commonProps = {
        ref: ref,
        whileHover: type === 'scale' ? { scale: scale.hover } : {},
        whileTap: type === 'scale' ? { scale: scale.tap } : {}
    }

    const [x, cycleX] = useCycle(directionValues[direction].translateX || 0, 0)
    const [y, cycleY] = useCycle(directionValues[direction].translateY || 0, 0)

    if (type === 'rotate') {
        return (
            <motion.div
                animate={{ rotate: 360 }}
                transition={{
                    repeat: Infinity,
                    repeatType: 'loop',
                    duration: 2,
                    repeatDelay: 0
                }}
                {...commonProps}
            >
                {children}
            </motion.div>
        )
    }

    const motionProps = {
        animate: {
            ...directionValues[direction],
            x: x,
            y: y
        },
        onHoverEnd: type === 'slide' ? cycleY : cycleX,
        onHoverStart: type === 'slide' ? cycleY : cycleX,
        ...commonProps
    }

    return (
        <motion.div {...motionProps}>
            {children}
        </motion.div>
    )
})

AnimateButton.propTypes = {
    children: PropTypes.node,
    offset: PropTypes.number,
    type: PropTypes.oneOf(['slide', 'scale', 'rotate']),
    direction: PropTypes.oneOf(['up', 'down', 'left', 'right']),
    scale: PropTypes.oneOfType([PropTypes.number, PropTypes.object])
}

export default AnimateButton

