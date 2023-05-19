// project imports
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'
import MinimalLayout from 'layout/MinimalLayout'

// canvas routing
const Canvas = Loadable(lazy(() => import('views/canvas')))
const MarketplaceCanvas = Loadable(lazy(() => import('views/marketplaces/MarketplaceCanvas')))

const canvasRoutes = [
  {
    path: '/canvas',
    element: <Canvas />
  },
  {
    path: '/canvas/:id',
    element: <Canvas />
  },
  {
    path: '/marketplace/:id',
    element: <MarketplaceCanvas />
  }
]

const CanvasRoutes = {
  path: '/',
  element: <MinimalLayout />,
  children: canvasRoutes
}

export default CanvasRoutes

