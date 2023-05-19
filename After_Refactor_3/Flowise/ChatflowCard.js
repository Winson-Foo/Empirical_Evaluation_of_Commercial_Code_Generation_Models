// material-ui
import { Card, CardContent, Grid } from '@mui/material'
import { RectangularSkeleton } from './SkeletonComponents'

// ==============================|| SKELETON - BRIDGE CARD ||============================== //

const ChatflowCard = () => (
    <Card>
        <CardContent>
            <Grid container direction='column'>
                <Grid item>
                    <Grid container justifyContent='space-between'>
                        <Grid item>
                            <RectangularSkeleton width={44} height={44} />
                        </Grid>
                        <Grid item>
                            <RectangularSkeleton width={34} height={34} />
                        </Grid>
                    </Grid>
                </Grid>
                <Grid item>
                    <RectangularSkeleton height={40} marginY={2} />
                </Grid>
                <Grid item>
                    <RectangularSkeleton height={30} />
                </Grid>
            </Grid>
        </CardContent>
    </Card>
)

export default ChatflowCard

// SkeletonComponents.js

import Skeleton from '@mui/material/Skeleton'

export const RectangularSkeleton = ({ height, width, marginY }) => (
    <Skeleton variant='rectangular' sx={{ my: marginY }} height={height} width={width} />
)

