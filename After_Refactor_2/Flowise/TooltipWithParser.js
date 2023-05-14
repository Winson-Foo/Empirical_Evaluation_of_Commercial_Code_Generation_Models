import { IconButton, Tooltip } from '@mui/material'
import PropTypes from 'prop-types'
import { useSelector } from 'react-redux'

export const TooltipWithParser = ({ title }) => {
    const { isDarkMode } = useSelector((state) => state.customization)

    return (
        <Tooltip title={title} placement='right'>
            <IconButton sx={{ height: 25, width: 25 }}>
                <span style={{ background: 'transparent', color: `${isDarkMode ? 'white' : 'inherit'}`, height: 18, width: 18 }}>i</span>
            </IconButton>
        </Tooltip>
    )
}

TooltipWithParser.propTypes = {
    title: PropTypes.node
}