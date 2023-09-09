// material-ui
import { Card, CardContent, Grid } from '@mui/material'
import Skeleton from '@mui/material/Skeleton'

// ==============================|| SKELETON - BRIDGE CARD ||============================== //

const ProfileImageSkeleton = () => (
    <Skeleton variant='rectangular' width={44} height={44} />
)

const IconSkeleton = () => (
    <Skeleton variant='rectangular' width={34} height={34} />
)

const TitleSkeleton = () => (
    <Skeleton variant='rectangular' sx={{ my: 2 }} height={40} />
)

const SubtitleSkeleton = () => (
    <Skeleton variant='rectangular' height={30} />
)

const ChatflowCard = () => (
    <Card>
        <CardContent>
            <Grid container direction='column'>
                <Grid item>
                    <Grid container justifyContent='space-between'>
                        <Grid item>
                            <ProfileImageSkeleton />
                        </Grid>
                        <Grid item>
                            <IconSkeleton />
                        </Grid>
                    </Grid>
                </Grid>
                <Grid item>
                    <TitleSkeleton />
                </Grid>
                <Grid item>
                    <SubtitleSkeleton />
                </Grid>
            </Grid>
        </CardContent>
    </Card>
)

export default ChatflowCard