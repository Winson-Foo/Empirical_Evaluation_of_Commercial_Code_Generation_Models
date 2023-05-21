// CardWrapper.tsx

import { styled } from '@mui/material/styles';
import MainCard from 'ui-component/cards/MainCard';

const CardWrapper = styled(MainCard)(({ theme }) => ({
  background: theme.palette.card.main,
  color: theme.darkTextPrimary,
  overflow: 'hidden',
  position: 'relative',
  boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
  cursor: 'pointer',

  '&:hover': {
    background: theme.palette.card.hover,
    boxShadow: '0 2px 14px 0 rgb(32 40 45 / 20%)',
  },
}));

export default CardWrapper;