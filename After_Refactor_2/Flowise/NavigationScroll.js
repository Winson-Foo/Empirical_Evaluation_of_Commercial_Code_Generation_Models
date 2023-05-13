import PropTypes from 'prop-types'
import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

/**
 * Component to scroll to the top of page when navigating to a new route.
 */
const NavigationScroll = ({ children }) => {
    const location = useLocation()
    const { pathname: currentPath } = location

    useEffect(() => {
        window.scrollTo({
            top: 0,
            left: 0,
            behavior: 'smooth'
        })
    }, [currentPath])

    return children || null
}

NavigationScroll.propTypes = {
    children: PropTypes.node
}

NavigationScroll.defaultProps = {
    children: null
}

export default NavigationScroll

