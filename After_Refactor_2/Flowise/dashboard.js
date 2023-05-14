// assets
import { IconHierarchy, IconBuildingStore, IconKey } from '@tabler/icons'

// constants
const icons = {
  IconHierarchy,
  IconBuildingStore,
  IconKey,
}

const chatflows = {
  id: 'chatflows',
  title: 'Chatflows',
  url: '/chatflows',
  icon: icons.IconHierarchy,
  breadcrumbs: true,
}

const marketplaces = {
  id: 'marketplaces',
  title: 'Marketplaces',
  url: '/marketplaces',
  icon: icons.IconBuildingStore,
  breadcrumbs: true,
}

const apikey = {
  id: 'apikey',
  title: 'API Keys',
  url: '/apikey',
  icon: icons.IconKey,
  breadcrumbs: true,
}

// dashboard menu items
const dashboard = {
  id: 'dashboard',
  title: '',
  type: 'group',
  children: [chatflows, marketplaces, apikey],
}

export default dashboard