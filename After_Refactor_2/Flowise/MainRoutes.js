import { lazy } from 'react'
import MainLayout from 'layout/MainLayout'
import Loadable from 'ui-component/loading/Loadable'

const VIEWS = {
  CHATFLOWS: lazy(() => import('views/chatflows')),
  MARKETPLACES: lazy(() => import('views/marketplaces')),
  APIKEY: lazy(() => import('views/apikey')),
}

const PATHS = {
  CHATFLOWS: '/',
  MARKETPLACES: '/marketplaces',
  APIKEY: '/apikey',
}

const createRoute = (path, view) => ({
  path: path,
  element: React.createElement(view),
})

const MainRoutes = {
  path: PATHS.CHATFLOWS,
  element: <MainLayout />,
  children: [
    createRoute(PATHS.CHATFLOWS, VIEWS.CHATFLOWS),
    createRoute(PATHS.MARKETPLACES, VIEWS.MARKETPLACES),
    createRoute(PATHS.APIKEY, VIEWS.APIKEY),
  ],
}

export default MainRoutes

