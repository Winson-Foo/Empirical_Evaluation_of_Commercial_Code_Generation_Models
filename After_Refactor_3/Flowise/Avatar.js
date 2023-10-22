import PropTypes from 'prop-types'
import { useTheme } from '@mui/material/styles'
import MuiAvatar from '@mui/material/Avatar'

const Avatar = ({ color = 'primary', outline = false, size = 'sm', sx, ...others }) => {
  const theme = useTheme()

  const sizes = {
    badge: theme.spacing(3.5),
    xs: theme.spacing(4.25),
    sm: theme.spacing(5),
    lg: theme.spacing(9),
    xl: theme.spacing(10.25),
    md: theme.spacing(7.5),
  }

  const { bgcolor, color: fontColor, borderColor } = {
    ...(color && !outline && { bgcolor: `${color}.main`, color: theme.palette.background.paper }),
    ...(outline && {
      color: color ? `${color}.main` : 'primary.main',
      bgcolor: theme.palette.background.paper,
      border: '2px solid',
      borderColor: color ? `${color}.main` : 'primary.main',
    }),
  }

  const sizeSX = {
    ...(size && sizes[size] ? { width: sizes[size], height: sizes[size] } : {}),
  }

  return <MuiAvatar sx={{ bgcolor, borderColor, color: fontColor, ...sizeSX, ...sx }} {...others} />
}

Avatar.propTypes = {
  className: PropTypes.string,
  color: PropTypes.string,
  outline: PropTypes.bool,
  size: PropTypes.string,
  sx: PropTypes.object,
}

export default Avatar