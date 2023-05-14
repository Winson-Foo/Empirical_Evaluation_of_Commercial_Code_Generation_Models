// ContractCard.js

import PropTypes from 'prop-types'
import styled from 'styled-components'
import { Box, Grid, Chip, Typography } from '@mui/material'
import MainCard from 'ui-component/cards/MainCard'
import SkeletonChatflowCard from 'ui-component/cards/Skeleton/ChatflowCard'

const CardWrapper = styled(MainCard)`
  background: ${({ theme }) => theme.palette.card.main};
  color: ${({ theme }) => theme.darkTextPrimary};
  overflow: hidden;
  position: relative;
  box-shadow: 0 2px 14px 0 rgb(32 40 45 / 8%);
  cursor: pointer;

  &:hover {
    background: ${({ theme }) => theme.palette.card.hover};
    box-shadow: 0 2px 14px 0 rgb(32 40 45 / 20%);
  }
`

const ActiveChatflowChip = styled(Chip)`
  height: 24px;
  padding: 0 6px;
  color: white;
  background-color: ${({ theme }) => theme.palette.success.dark};
`

const ContractCard = ({ isLoading, data, images, onClick }) => {
  const { palette } = useTheme()

  return (
    <>
      {isLoading ? (
        <SkeletonChatflowCard />
      ) : (
        <CardWrapper border={false} content={false} onClick={onClick}>
          <Box padding={2.25}>
            <Grid container direction='column'>
              <div>
                <Typography variant='h5' fontWeight='500'>
                  {data.name}
                </Typography>
              </div>
              {data.description && <Typography variant='subtitle1'>{data.description}</Typography>}
              <Grid sx={{ mt: 1, mb: 1 }} container direction='row'>
                {data.deployed && (
                  <Grid item>
                    <ActiveChatflowChip label='Deployed' />
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
                        backgroundColor: 'white',
                      }}
                    >
                      <img style={{ width: '100%', height: '100%', padding: 5, objectFit: 'contain' }} alt='' src={img} />
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

ContractCard.propTypes = {
  isLoading: PropTypes.bool,
  data: PropTypes.object,
  images: PropTypes.array,
  onClick: PropTypes.func,
}

export default ContractCard

