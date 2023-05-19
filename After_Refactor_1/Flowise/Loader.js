// styles/LoaderStyles.js
import { styled } from '@mui/material/styles'

export const LoaderWrapper = styled('div')({
  position: 'fixed',
  top: 0,
  left: 0,
  zIndex: 1301,
  width: '100%'
})

// components/Loader.js
import LinearProgress from '@mui/material/LinearProgress'
import { LoaderWrapper } from '../styles/LoaderStyles'

const Loader = () => (
  <LoaderWrapper>
    <LinearProgress color="primary" />
  </LoaderWrapper>
)

export default Loader