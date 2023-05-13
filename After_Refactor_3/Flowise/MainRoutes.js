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

// ChatflowsRoutes.js
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'

const Chatflows = Loadable(lazy(() => import('views/chatflows')))

const ChatflowsRoutes = [
    {
        path: '/',
        element: <Chatflows />
    },
    {
        path: '/chatflows',
        element: <Chatflows />
    },
]

export default ChatflowsRoutes

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

// APIKeyRoutes.js
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'

const APIKey = Loadable(lazy(() => import('views/apikey')))

const APIKeyRoutes = [
    {
        path: '/apikey',
        element: <APIKey />
    },
]

export default APIKeyRoutes