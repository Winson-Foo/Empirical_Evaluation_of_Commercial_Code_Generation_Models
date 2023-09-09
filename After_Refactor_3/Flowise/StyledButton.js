import { styled } from '@mui/material/styles';
import { Button } from '@mui/material';

// Define the style object function for use in multiple components
const buttonStyle = ({ theme, color = 'primary' }) => ({
  color: 'white',
  backgroundColor: theme.palette[color].main,
  '&:hover': {
    backgroundColor: theme.palette[color].main,
    backgroundImage: `linear-gradient(rgb(0 0 0/10%) 0 0)`,
  },
});

// Create the styled component using the style object function
const StyledButton = styled(Button)(buttonStyle);

export default StyledButton; // Export the component for use in other files.

