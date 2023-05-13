// LoaderWrapper.styles.js
import { styled } from '@mui/material/styles'

const LoaderWrapper = styled('div')({
    position: 'fixed',
    top: 0,
    left: 0,
    zIndex: 1301,
    width: '100%'
})

export default LoaderWrapper

// Loader.js
import LinearProgress from '@mui/material/LinearProgress'
import LoaderWrapper from './LoaderWrapper.styles'

const Loader = ({ color = 'primary' }) => (
    <LoaderWrapper>
        <LinearProgress color={color} />
    </LoaderWrapper>
)

export default Loader