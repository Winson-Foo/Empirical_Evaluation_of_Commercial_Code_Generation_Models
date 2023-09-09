// routes.js
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'
import MinimalLayout from 'layout/MinimalLayout'

const Canvas = Loadable(lazy(() => import('views/canvas')))
const MarketplaceCanvas = Loadable(lazy(() => import('views/marketplaces/MarketplaceCanvas')))

export const PATHS = {
    canvas: '/canvas',
    canvasID: '/canvas/:id',
    marketplaceID: '/marketplace/:id',
}

export const CANVAS_ROUTES = {
    path: '/',
    element: <MinimalLayout />,
    children: [
        {
            path: PATHS.canvas,
            element: <Canvas />
        },
        {
            path: PATHS.canvasID,
            element: <Canvas />
        },
        {
            path: PATHS.marketplaceID,
            element: <MarketplaceCanvas />
        }
    ]
}

// index.js
import { CANVAS_ROUTES } from './routes'

export { CANVAS_ROUTES } // named export of the routes constant

// usage in other files
import { CANVAS_ROUTES } from 'path/to/index'
// use it in <Routes> or <BrowserRouter> as {CANVAS_ROUTES} instead of directly using the object.

