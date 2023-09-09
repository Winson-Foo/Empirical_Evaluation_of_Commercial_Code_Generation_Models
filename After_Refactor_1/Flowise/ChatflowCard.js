// material-ui
import { Card, CardContent, Grid } from '@mui/material'
import Skeleton from '@mui/material/Skeleton'

// ==================================|| SKELETON COMPONENTS ||================================= //

const AvatarSkeleton = () => (
  <Skeleton variant='rectangular' width={44} height={44} />
)

const IconSkeleton = () => (
  <Skeleton variant='rectangular' width={34} height={34} />
)

const TextSkeleton = () => (
  <Skeleton variant='rectangular' sx={{ my: 2 }} height={40} />
)

const SubtextSkeleton = () => (
  <Skeleton variant='rectangular' height={30} />
)

// ==============================|| SKELETON - BRIDGE CARD ||============================== //

const ChatflowCard = () => (
  <Card>
    <CardContent>
      <Grid container direction='column'>
        <Grid item>
          <Grid container justifyContent='space-between'>
            <Grid item>
              <AvatarSkeleton />
            </Grid>
            <Grid item>
              <IconSkeleton />
            </Grid>
          </Grid>
        </Grid>
        <Grid item>
          <TextSkeleton />
        </Grid>
        <Grid item>
          <SubtextSkeleton />
        </Grid>
      </Grid>
    </CardContent>
  </Card>
)

export default ChatflowCard

