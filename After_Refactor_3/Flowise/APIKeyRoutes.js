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