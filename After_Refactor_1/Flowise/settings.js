// assets
import * as icons from '@tabler/icons'

// constants
const SETTINGS_MENU_ITEMS = [
  { id: 'duplicateChatflow', title: 'Duplicate Chatflow', icon: icons.IconCopy },
  { id: 'loadChatflow', title: 'Load Chatflow', icon: icons.IconFileUpload },
  { id: 'exportChatflow', title: 'Export Chatflow', icon: icons.IconFileExport },
  { id: 'deleteChatflow', title: 'Delete Chatflow', icon: icons.IconTrash }
]
const SETTINGS_MENU = { id: 'settings', title: '', type: 'group', children: [] }

// functions
function buildMenuItem(item) {
  return {
    id: item.id,
    title: item.title,
    type: 'item',
    url: '',
    icon: item.icon
  }
}

function buildSettingsMenu() {
  SETTINGS_MENU.children = SETTINGS_MENU_ITEMS.map(buildMenuItem)
  return SETTINGS_MENU
}

// default export
export default buildSettingsMenu()

