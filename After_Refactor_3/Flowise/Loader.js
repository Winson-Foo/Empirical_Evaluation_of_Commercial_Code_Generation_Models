// Loader.js
import LinearProgress from '@mui/material/LinearProgress'
import LoaderWrapper from './LoaderWrapper.styles'

const Loader = ({ color = 'primary' }) => (
    <LoaderWrapper>
        <LinearProgress color={color} />
    </LoaderWrapper>
)

export default Loader