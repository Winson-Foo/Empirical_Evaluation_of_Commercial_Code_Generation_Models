//==============================|| ANIMATION COMPONENT ||==============================//

import { motion } from 'framer-motion'
import PropTypes from 'prop-types'

const MotionComponent = ({ children, variants, initial, animate, exit, transition }) => (
  <motion.div initial={initial} animate={animate} exit={exit} variants={variants} transition={transition}>
    {children}
  </motion.div>
)

MotionComponent.propTypes = {
  children: PropTypes.node,
  variants: PropTypes.object,
  initial: PropTypes.string,
  animate: PropTypes.string,
  exit: PropTypes.string,
  transition: PropTypes.object
}

export default MotionComponent

// ==============================|| NAVIGATION ANIMATION ||============================== //

import PropTypes from 'prop-types'
import MotionComponent from './MotionComponent'

const NavMotion = ({ children }) => {
  const motionVariants = {
    initial: {
      opacity: 0,
      scale: 0.99
    },
    in: {
      opacity: 1,
      scale: 1
    },
    out: {
      opacity: 0,
      scale: 1.01
    }
  }

  const motionTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.4
  }

  return (
    <MotionComponent initial='initial' animate='in' exit='out' variants={motionVariants} transition={motionTransition}>
      {children}
    </MotionComponent>
  )
}

NavMotion.propTypes = {
  children: PropTypes.node
}

export default NavMotion