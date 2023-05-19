import { Box, Grid, Chip, Typography } from '@mui/material'
import { styled, useTheme } from '@mui/material/styles'
import PropTypes from 'prop-types'
import MainCard from 'ui-component/cards/MainCard'
import SkeletonChatflowCard from 'ui-component/cards/Skeleton/ChatflowCard'

const CardWrapper = styled(MainCard)(({ theme }) => ({
  background: theme.palette.card.main,
  color: theme.darkTextPrimary,
  overflow: 'hidden',
  position: 'relative',
  boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
  cursor: 'pointer',
  '&:hover': {
    background: theme.palette.card.hover,
    boxShadow: '0 2px 14px 0 rgb(32 40 45 / 20%)'
  }
}))

const ItemCard = ({ isLoading, data: { name, description, deployed }, images, onClick }) => {
  const theme = useTheme()

  const chipStyle = {
    height: 24,
    padding: '0 6px'
  }

  const activeChipStyle = {
    ...chipStyle,
    color: 'white',
    backgroundColor: theme.palette.success.dark
  }

  return (
    <>{isLoading ? (
      <SkeletonChatflowCard />
    ) : (
      <CardWrapper border={false} content={false} onClick={onClick}>
        <Box sx={{ p: 2.25 }}>
          <Grid container direction='column'>
            <div>
              <Typography sx={{ fontSize: '1.5rem', fontWeight: 500 }}>{name}</Typography>
            </div>
            {description && <span style={{ marginTop: 10 }}>{description}</span>}
            <Grid sx={{ mt: 1, mb: 1 }} container direction='row'>
              {deployed && (
                <Grid item>
                  <Chip label='Deployed' sx={activeChipStyle} />
                </Grid>
              )}
            </Grid>
            {images && (
              <div style={{ display: 'flex', flexDirection: 'row', marginTop: 10 }}>
                {images.map((img) => (
                  <div
                    key={img}
                    style={{
                      width: 40,
                      height: 40,
                      marginRight: 5,
                      borderRadius: '50%',
                      backgroundColor: 'white'
                    }}
                  >
                    <img
                      style={{ width: '100%', height: '100%', padding: 5, objectFit: 'contain' }}
                      alt=''
                      src={img}
                    />
                  </div>
                ))}
              </div>
            )}
          </Grid>
        </Box>
      </CardWrapper>
    )}
    </>
  )
}

ItemCard.propTypes = {
  isLoading: PropTypes.bool,
  data: PropTypes.object,
  images: PropTypes.array,
  onClick: PropTypes.func
}

export default ItemCard