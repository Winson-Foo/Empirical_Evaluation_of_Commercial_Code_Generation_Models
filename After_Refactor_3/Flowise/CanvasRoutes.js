// components
import { lazy } from 'react';
import Loadable from 'ui-component/loading/Loadable';
import MinimalLayout from 'layout/MinimalLayout';

// constants
const CANVAS_PATH = '/canvas';
const MARKETPLACE_PATH = '/marketplace';
const CANVAS_COMPONENT = lazy(() => import('views/canvas'));
const MARKETPLACE_COMPONENT = lazy(() => import('views/marketplaces/MarketplaceCanvas'));

// routing
const routes = [
  {
    path: CANVAS_PATH,
    element: <Canvas />
  },
  {
    path: `${CANVAS_PATH}/:id`,
    element: <Canvas />
  },
  {
    path: `${MARKETPLACE_PATH}/:id`,
    element: <MarketplaceCanvas />
  }
];

const CanvasRoutes = {
  path: '/',
  element: <MinimalLayout />,
  children: routes
};

export default CanvasRoutes;

