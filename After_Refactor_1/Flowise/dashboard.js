// assets
import { IconHierarchy, IconBuildingStore, IconKey } from '@tabler/icons';

// constant
const icons = {
  IconHierarchy,
  IconBuildingStore,
  IconKey,
};

// dashboard menu items
const dashboardMenuItems = [
  {
    id: 'chatflows',
    title: 'Chatflows',
    url: '/chatflows',
    icon: icons.IconHierarchy,
    breadcrumbs: true,
  },
  {
    id: 'marketplaces',
    title: 'Marketplaces',
    url: '/marketplaces',
    icon: icons.IconBuildingStore,
    breadcrumbs: true,
  },
  {
    id: 'apikey',
    title: 'API Keys',
    url: '/apikey',
    icon: icons.IconKey,
    breadcrumbs: true,
  },
];

// dashboard group item
const dashboardGroupItem = {
  id: 'dashboard',
  title: '',
  type: 'group',
  children: dashboardMenuItems,
};

export default dashboardGroupItem;