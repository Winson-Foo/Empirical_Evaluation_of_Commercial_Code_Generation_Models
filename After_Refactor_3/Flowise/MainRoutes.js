// MainRoutes.js
import { lazy } from 'react'
import MainLayout from 'layout/MainLayout'
import Loadable from 'ui-component/loading/Loadable'

import ChatflowsRoutes from './ChatflowsRoutes'
import MarketplacesRoutes from './MarketplacesRoutes'
import APIKeyRoutes from './APIKeyRoutes'

const MainRoutes = {
    path: '/',
    element: <MainLayout />,
    children: [
        ...ChatflowsRoutes,
        ...MarketplacesRoutes,
        ...APIKeyRoutes
    ]
}

export default MainRoutes