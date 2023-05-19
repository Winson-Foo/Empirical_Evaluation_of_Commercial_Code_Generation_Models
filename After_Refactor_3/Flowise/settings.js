// assets/icons.js
import { IconTrash, IconFileUpload, IconFileExport, IconCopy } from '@tabler/icons'

export const TRASH_ICON = IconTrash;
export const UPLOAD_ICON = IconFileUpload;
export const EXPORT_ICON = IconFileExport;
export const COPY_ICON = IconCopy;

// constants/settings.js
import { TRASH_ICON, UPLOAD_ICON, EXPORT_ICON, COPY_ICON } from '../assets/icons';

const CHATFLOW_SETTINGS = {
    id: 'settings',
    title: '',
    type: 'group',
    children: [
        {
            id: 'duplicateChatflow',
            title: 'Duplicate Chatflow',
            type: 'item',
            url: '',
            icon: COPY_ICON
        },
        {
            id: 'loadChatflow',
            title: 'Load Chatflow',
            type: 'item',
            url: '',
            icon: UPLOAD_ICON
        },
        {
            id: 'exportChatflow',
            title: 'Export Chatflow',
            type: 'item',
            url: '',
            icon: EXPORT_ICON
        },
        {
            id: 'deleteChatflow',
            title: 'Delete Chatflow',
            type: 'item',
            url: '',
            icon: TRASH_ICON
        }
    ]
};

export default CHATFLOW_SETTINGS;

