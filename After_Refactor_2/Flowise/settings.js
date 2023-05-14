// assets
import { IconTrash, IconFileUpload, IconFileExport, IconCopy } from '@tabler/icons'

// constant data
const MENU_ITEMS = [
    {
        id: 'duplicateChatflow',
        title: 'Duplicate Chatflow',
        icon: IconCopy
    },
    {
        id: 'loadChatflow',
        title: 'Load Chatflow',
        icon: IconFileUpload
    },
    {
        id: 'exportChatflow',
        title: 'Export Chatflow',
        icon: IconFileExport
    },
    {
        id: 'deleteChatflow',
        title: 'Delete Chatflow',
        icon: IconTrash
    }
];

// settings data using MENU_ITEMS
const settings = {
    id: 'settings',
    title: '',
    type: 'group',
    children: MENU_ITEMS.map(item => ({
        ...item,
        type: 'item',
        url: '',
    }))
}

export default settings