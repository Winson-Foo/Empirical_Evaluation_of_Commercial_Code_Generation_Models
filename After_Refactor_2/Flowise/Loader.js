// material-ui
import LinearProgress from '@mui/material/LinearProgress'
import { styled } from '@mui/material/styles'

// styles
const LoaderWrapper = styled('div')({
    position: 'fixed',
    top: 0,
    left: 0,
    zIndex: 1301,
    width: '100%'
})

// component
const LoaderComponent = ({ color }) => (
    <LoaderWrapper>
        <LinearProgress color={color} />
    </LoaderWrapper>
)

// default props
LoaderComponent.defaultProps = {
    color: 'primary'
}

export default LoaderComponent