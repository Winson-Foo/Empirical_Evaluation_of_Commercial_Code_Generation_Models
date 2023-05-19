import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import MuiAvatar from '@mui/material/Avatar';

const AvatarSizes = {
  badge: [3.5, 3.5],
  xs: [4.25, 4.25],
  sm: [5, 5],
  lg: [9, 9],
  xl: [10.25, 10.25],
  md: [7.5, 7.5],
};

const Avatar = ({ color, outline, size = 'md', sx, ...others }) => {
  const theme = useTheme();
  const [width, height] = AvatarSizes[size] || [];

  const colorSX = color && !outline && {
    color: theme.palette.background.paper,
    bgcolor: `${color}.main`
  };

  const outlineSX = outline && {
    color: color ? `${color}.main` : `primary.main`,
    bgcolor: theme.palette.background.paper,
    border: '2px solid',
    borderColor: color ? `${color}.main` : `primary.main`
  };

  const sizeSX = width && height && {
    width: theme.spacing(width),
    height: theme.spacing(height)
  };

  return (
    <MuiAvatar
      sx={{ ...colorSX, ...outlineSX, ...sizeSX, ...sx }}
      {...others}
    />
  );
};

Avatar.propTypes = {
  className: PropTypes.string,
  color: PropTypes.string,
  outline: PropTypes.bool,
  size: PropTypes.string,
  sx: PropTypes.object
};

export default Avatar;