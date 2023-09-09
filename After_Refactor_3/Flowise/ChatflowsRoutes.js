// ChatflowsRoutes.js
import { lazy } from 'react'
import Loadable from 'ui-component/loading/Loadable'

const Chatflows = Loadable(lazy(() => import('views/chatflows')))

const ChatflowsRoutes = [
    {
        path: '/',
        element: <Chatflows />
    },
    {
        path: '/chatflows',
        element: <Chatflows />
    },
]

export default ChatflowsRoutes