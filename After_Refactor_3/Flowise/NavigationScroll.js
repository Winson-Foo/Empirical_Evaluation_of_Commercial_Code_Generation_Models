import PropTypes from 'prop-types';
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Scroll to top of page on navigation change
 * @component
 * @param {node} children - Child components to render
 */
const ScrollToTop = ({ children }) => {
  const location = useLocation();
  const { pathname } = location;

  /**
   * Scroll to top of page with a smooth animation
   * @function
   * @param {string} behavior - Scroll behavior
   */
  const scrollToTop = (behavior) => {
    window.scrollTo({
      top: 0,
      left: 0,
      behavior,
    });
  };

  useEffect(() => {
    scrollToTop('smooth');
  }, [pathname]);

  return children || null;
};

ScrollToTop.propTypes = {
  children: PropTypes.node,
};

export default ScrollToTop;

