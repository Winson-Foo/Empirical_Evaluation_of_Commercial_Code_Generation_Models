// styles.ts

import { makeStyles } from '@mui/styles';

export const useCardStyles = makeStyles((theme) => ({
  card: {
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
  },
  title: {
    fontSize: '1.5rem',
    fontWeight: 500,
    marginBottom: theme.spacing(1),
  },
  description: {
    marginTop: theme.spacing(1),
  },
  chip: {
    height: 24,
    padding: '0 6px',
    color: '#fff',
    backgroundColor: theme.palette.success.dark,
  },
  imageContainer: {
    display: 'flex',
    flexDirection: 'row',
    marginTop: theme.spacing(1),
  },
  imageWrapper: {
    width: 40,
    height: 40,
    marginRight: theme.spacing(1),
    borderRadius: '50%',
    backgroundColor: 'white',
  },
  image: {
    width: '100%',
    height: '100%',
    padding: theme.spacing(1),
    objectFit: 'contain',
  },
}));