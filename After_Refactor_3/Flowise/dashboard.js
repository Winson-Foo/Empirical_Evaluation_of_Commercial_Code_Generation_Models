// assets
import { IconHierarchy, IconBuildingStore, IconKey } from '@tabler/icons'

// function to generate menu items
const generateMenuItems = (items) => {
  // define icons object within function
  const icons = { IconHierarchy, IconBuildingStore, IconKey }
  
  return items.map(({ id, title, type, url, icon, breadcrumbs }) => {
    return {
      id,
      title,
      type,
      url,
      icon: icons[icon],
      breadcrumbs
    }
  })
}

// define menu item objects
const chatflows = {
  id: 'chatflows',
  title: 'Chatflows',
  type: 'item',
  url: '/chatflows',
  icon: 'IconHierarchy',
  breadcrumbs: true
}

const marketplaces = {
  id: 'marketplaces',
  title: 'Marketplaces',
  type: 'item',
  url: '/marketplaces',
  icon: 'IconBuildingStore',
  breadcrumbs: true
}

const apikey = {
  id: 'apikey',
  title: 'API Keys',
  type: 'item',
  url: '/apikey',
  icon: 'IconKey',
  breadcrumbs: true
}

// generate menu items
const dashboardItems = generateMenuItems([chatflows, marketplaces, apikey])

// define dashboard object
const dashboard = {
  id: 'dashboard',
  title: '',
  type: 'group',
  children: dashboardItems
}

export default dashboard