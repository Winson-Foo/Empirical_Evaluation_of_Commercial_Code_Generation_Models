import React from 'react';
import { Route } from 'react-router-dom';
import MainLayout from 'layout/MainLayout';
import Loadable from 'ui-component/loading/Loadable';

const Chatflows = Loadable(() => import('views/chatflows'));
const Marketplaces = Loadable(() => import('views/marketplaces'));
const APIKey = Loadable(() => import('views/apikey'));

const MainRoutes = () => {
  return (
    <MainLayout>
      <Route exact path="/" component={Chatflows} />
      <Route exact path="/chatflows" component={Chatflows} />
      <Route exact path="/marketplaces" component={Marketplaces} />
      <Route exact path="/apikey" component={APIKey} />
    </MainLayout>
  );
};

export default MainRoutes;

