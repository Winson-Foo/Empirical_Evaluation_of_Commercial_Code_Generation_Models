import { styled } from '@mui/material/styles'
import { Fab } from '@mui/material'

const FAB_BACKGROUND_COLOR = 'primary'
const FAB_HOVER_BACKGROUND_COLOR = `linear-gradient(rgb(0 0 0/10%) 0 0)`

const StyledFab = styled(Fab)(({ theme, color = FAB_BACKGROUND_COLOR }) => ({
    color: 'white',
    backgroundColor: theme.palette[color].main,
    '&:hover': {
        backgroundColor: theme.palette[color].main,
        backgroundImage: FAB_HOVER_BACKGROUND_COLOR,
    },
}))

export default StyledFab

