// MarketplacesRoutes.js
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'

const Marketplaces = Loadable(lazy(() => import('views/marketplaces')))

const MarketplacesRoutes = [
    {
        path: '/marketplaces',
        element: <Marketplaces />
    },
]

export default MarketplacesRoutes