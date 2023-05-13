// ItemCard.tsx

import PropTypes from 'prop-types';
import { Box, Grid, Chip, Typography } from '@mui/material';
import MainCard from 'ui-component/cards/MainCard';
import SkeletonChatflowCard from 'ui-component/cards/Skeleton/ChatflowCard';
import { CardData } from '../interfaces';
import { useCardStyles } from '../styles';

interface Props {
  isLoading?: boolean;
  data?: CardData;
  images?: string[];
  onClick?: () => void;
}

const ItemCard = ({ isLoading = false, data = {}, images = [], onClick }: Props) => {
  const classes = useCardStyles();

  if (isLoading) {
    return <SkeletonChatflowCard />;
  }

  return (
    <MainCard border={false} content={false} onClick={onClick} className={classes.card}>
      <Box sx={{ p: 2.25 }}>
        <Grid container direction="column">
          <Typography className={classes.title} variant="h5">
            {data.name}
          </Typography>
          {data.description && <span className={classes.description}>{data.description}</span>}
          <Grid sx={{ mt: 1, mb: 1 }} container direction="row">
            {data.deployed && (
              <Grid item>
                <Chip className={classes.chip} label="Deployed" color="success" />
              </Grid>
            )}
          </Grid>
          {images.length > 0 && (
            <div className={classes.imageContainer}>
              {images.map((img) => (
                <div key={img} className={classes.imageWrapper}>
                  <img className={classes.image} alt="" src={img} />
                </div>
              ))}
            </div>
          )}
        </Grid>
      </Box>
    </MainCard>
  );
};

ItemCard.propTypes = {
  isLoading: PropTypes.bool,
  data: PropTypes.shape({
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    deployed: PropTypes.bool.isRequired,
  }).isRequired,
  images: PropTypes.arrayOf(PropTypes.string),
  onClick: PropTypes.func,
};

export default ItemCard;

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

// interfaces.ts

export interface CardData {
  name: string;
  description?: string;
  deployed: boolean;
} 

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