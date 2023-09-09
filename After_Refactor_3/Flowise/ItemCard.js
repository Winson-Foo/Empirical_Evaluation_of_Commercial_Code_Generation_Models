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