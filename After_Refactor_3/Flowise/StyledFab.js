import { styled } from '@mui/material/styles';
import { Fab } from '@mui/material';

const buttonStyles = ({ theme, color }) => ({
  color: 'white',
  backgroundColor: theme.palette[color].main,
  '&:hover': {
    backgroundColor: theme.palette[color].main,
    backgroundImage: `linear-gradient(rgb(0 0 0/10%) 0 0)`,
  },
});

const StyledFab = styled(Fab)(buttonStyles);

StyledFab.defaultProps = {
  color: 'primary',
};

export default StyledFab;

