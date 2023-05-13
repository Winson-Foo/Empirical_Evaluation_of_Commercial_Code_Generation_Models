import PropTypes from 'prop-types'
import { forwardRef } from 'react'
// third-party
import { motion, useCycle } from 'framer-motion'

const DIRECTION_OFFSETS = {
  up: { offset1: 10, offset2: 0 },
  down: { offset1: 0, offset2: 10 },
  left: { offset1: 10, offset2: 0 },
  right: { offset1: 0, offset2: 10 },
}

const AnimateButton = forwardRef(function AnimateButton({
  children,
  type = 'scale',
  direction = 'right',
  scale = { hover: 1, tap: 0.9 },
  offset = 10,
}, ref) {
  const { offset1, offset2 } = DIRECTION_OFFSETS[direction]

  const [x, cycleX] = useCycle(offset1, offset2)
  const [y, cycleY] = useCycle(offset1, offset2)

  const animateRotation = () => {
    return (
      <motion.div
        ref={ref}
        animate={{ rotate: 360 }}
        transition={{
          repeat: Infinity,
          repeatType: 'loop',
          duration: 2,
          repeatDelay: 0,
        }}>
        {children}
      </motion.div>
    )
  }

  const animateSlide = () => {
    const motionProps = {
      ref,
      animate: {
        x: direction === 'left' || direction === 'right' ? x : '',
        y: direction === 'up' || direction === 'down' ? y : '',
      },
      onHoverEnd: direction === 'up' || direction === 'down' ? cycleY : cycleX,
      onHoverStart: direction === 'up' || direction === 'down' ? cycleY : cycleX,
    }
    return <motion.div {...motionProps}>{children}</motion.div>
  }

  const animateScale = () => {
    const motionProps = {
      ref,
      whileHover: { scale: scale.hover },
      whileTap: { scale: scale.tap },
    }
    return <motion.div {...motionProps}>{children}</motion.div>
  }

  switch (type) {
    case 'rotate':
      return animateRotation()
    case 'slide':
      return animateSlide()
    case 'scale':
    default:
      return animateScale()
  }
})

AnimateButton.propTypes = {
  children: PropTypes.node,
  offset: PropTypes.number,
  type: PropTypes.oneOf(['slide', 'scale', 'rotate']),
  direction: PropTypes.oneOf(['up', 'down', 'left', 'right']),
  scale: PropTypes.oneOfType([PropTypes.number, PropTypes.object]),
}

export default AnimateButton

