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