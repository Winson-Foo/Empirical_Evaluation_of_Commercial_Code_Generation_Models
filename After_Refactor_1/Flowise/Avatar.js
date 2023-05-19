// Avatar.jsx

import PropTypes from 'prop-types'
import { useTheme } from '@mui/material/styles'
import MuiAvatar from '@mui/material/Avatar'

import { getAvatarColorSX, getAvatarOutlineSX, getAvatarSizeSX } from './helpers/avatar'

const Avatar = ({ color, outline, size, sx, ...others }) => {
  const theme = useTheme()

  const colorSX = getAvatarColorSX(theme, color, outline)
  const outlineSX = getAvatarOutlineSX(theme, outline, color)
  const sizeSX = getAvatarSizeSX(theme, size)

  return <MuiAvatar sx={{ ...colorSX, ...outlineSX, ...sizeSX, ...sx }} {...others} />
}

Avatar.propTypes = {
  className: PropTypes.string,
  color: PropTypes.string,
  outline: PropTypes.bool,
  size: PropTypes.oneOf(['badge', 'xs', 'sm', 'lg', 'xl', 'md']),
  sx: PropTypes.object
}

export default Avatar


// helpers/avatar.js

const COLORS = {
  primary: 'primary.main',
  secondary: 'secondary.main',
  success: 'success.main',
  info: 'info.main',
  warning: 'warning.main',
  error: 'error.main'
}

const SIZES = {
  badge: {
    width: 21,
    height: 21
  },
  xs: {
    width: 26,
    height: 26
  },
  sm: {
    width: 32,
    height: 32
  },
  lg: {
    width: 58,
    height: 58
  },
  xl: {
    width: 68,
    height: 68
  },
  md: {
    width: 48,
    height: 48
  }
}

export const getAvatarColorSX = (theme, color, outline) => {
  if (outline) return {}
  const bgColor = color ? COLORS[color] : COLORS.primary
  return { color: theme.palette.background.paper, bgcolor: bgColor }
}

export const getAvatarOutlineSX = (theme, outline, color) => {
  if (!outline) return {}
  const borderColor = color ? COLORS[color] : COLORS.primary
  return {
    color: borderColor,
    bgcolor: theme.palette.background.paper,
    border: '2px solid',
    borderColor
  }
}

export const getAvatarSizeSX = (theme, size) => {
  const { width, height } = SIZES[size] || {}
  return { width: theme.spacing(width), height: theme.spacing(height) }
}

