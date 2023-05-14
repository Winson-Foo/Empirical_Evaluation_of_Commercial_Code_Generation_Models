import { styled } from '@mui/material/styles';
import { Button } from '@mui/material';

// Define a function to create a styled button.
const createStyledButton = ({ theme, color = 'primary' }) => ({
  color: 'white',
  backgroundColor: theme.palette[color].main,

  // Add a hover effect using the theme's color palette.
  '&:hover': {
    backgroundColor: theme.palette[color].main,
    backgroundImage: `linear-gradient(rgb(0 0 0/10%) 0 0)`,
  },
});

// Create the styled button component using MUI's `styled` function.
const StyledButton = styled(Button)(createStyledButton);

export default StyledButton; // Export the component.

