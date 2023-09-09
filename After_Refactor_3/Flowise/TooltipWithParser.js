import { Info } from '@mui/icons-material'
import { IconButton, Tooltip } from '@mui/material'
import parser from 'html-react-parser'
import PropTypes from 'prop-types'
import { useSelector } from 'react-redux'

const StyledIconButton = ({ darkMode, children }) => (
  <IconButton sx={{ height: 25, width: 25 }}>
    {children}
    <style jsx>{`
      svg {
        background: transparent;
        color: ${darkMode ? 'white' : 'inherit'};
        height: 18px;
        width: 18px;
      }
    `}</style>
  </IconButton>
)

StyledIconButton.propTypes = {
  darkMode: PropTypes.bool.isRequired,
  children: PropTypes.node.isRequired,
}

const TooltipWithParser = ({ title }) => {
  const customization = useSelector((state) => state.customization)

  return (
    <Tooltip title={parser(title)} placement='right'>
      <StyledIconButton darkMode={customization.isDarkMode}>
        <Info />
      </StyledIconButton>
    </Tooltip>
  )
}

TooltipWithParser.propTypes = {
  title: PropTypes.node.isRequired,
}

export default TooltipWithParser