// styledButton.js
import { styled } from '@mui/material/styles';
import { Button } from '@mui/material';
import { COLOR_PRIMARY } from './constants';

export const StyledButton = styled(Button)(({ theme, color = COLOR_PRIMARY }) => ({
  color: 'white',
  backgroundColor: theme.palette[color].main,
  '&:hover': {
    backgroundColor: theme.palette[color].main,
    backgroundImage: `linear-gradient(rgb(0 0 0/10%) 0 0)`,
  },
}));
