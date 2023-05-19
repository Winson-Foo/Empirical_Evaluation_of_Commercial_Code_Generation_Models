import PropTypes from 'prop-types';
import { motion } from 'framer-motion';

const NavMotion = ({ children }) => {
  const navMotionVariants = {
    initial: { opacity: 0, scale: 0.99 },
    in: { opacity: 1, scale: 1 },
    out: { opacity: 0, scale: 1.01 },
  };

  const navMotionTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.4,
  };

  return (
    <motion.div
      initial="initial"
      animate="in"
      exit="out"
      variants={navMotionVariants}
      transition={navMotionTransition}
    >
      {children}
    </motion.div>
  );
};

NavMotion.propTypes = {
  children: PropTypes.node,
};

export default NavMotion;

