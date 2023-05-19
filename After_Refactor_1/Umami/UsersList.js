// useUsersApi.js

import { useApi } from 'hooks/useApi'
import useUser from 'hooks/useUser'

export default function useUsersApi() {
  const { user } = useUser()
  const { get, useQuery } = useApi()
  const { data, isLoading, error, refetch } = useQuery(['user'], () => get(`/users`), {
    enabled: !!user,
  })

  return { data, isLoading, error, refetch }
}

// messages.js

import { defineMessages } from 'react-intl'

const messages = defineMessages({
  saved: {
    id: 'users.saved',
    defaultMessage: 'Saved successfully',
  },
  userDeleted: {
    id: 'users.userDeleted',
    defaultMessage: 'User deleted successfully',
  },
  noUsers: {
    id: 'users.noUsers',
    defaultMessage: 'No users found',
  },
  // add more messages here as needed
})

export default messages

// useMessages.js

import { useIntl } from 'react-intl'
import messages from './messages'

export default function useMessages() {
  const { formatMessage } = useIntl()
  const labels = { users: formatMessage({ id: 'users.title', defaultMessage: 'Users' })}
  
  return { formatMessage, labels, messages }
}

// PageHeader.js

import UserAddButton from '../UserAddButton'

export default function PageHeader({ title, onSave }) {
  return (
    <div>
      <h1>{title}</h1>
      <UserAddButton onSave={onSave} />
    </div>
  )
}

// UsersList.js

import { useToast } from 'react-basics'
import Page from 'components/layout/Page'
import PageHeader from 'components/layout/PageHeader'
import EmptyPlaceholder from 'components/common/EmptyPlaceholder'
import UsersTable from './UsersTable'
import UserAddButton from './UserAddButton'
import useUsersApi from './useUsersApi'
import useMessages from 'hooks/useMessages'

export function UsersList() {
  const { data, isLoading, error, refetch } = useUsersApi()
  const { formatMessage, labels, messages } = useMessages()
  const { toast, showToast } = useToast()
  const hasData = data && data.length !== 0

  const handleSave = () => {
    refetch().then(() => showToast({ message: formatMessage(messages.saved), variant: 'success' }))
  }

  const handleDelete = () => {
    refetch().then(() =>
      showToast({ message: formatMessage(messages.userDeleted), variant: 'success' }),
    )
  }

  return (
    <Page loading={isLoading} error={error}>
      {toast}
      <PageHeader title={formatMessage(labels.users)} onSave={handleSave} />
      {hasData && <UsersTable data={data} onDelete={handleDelete} />}
      {!hasData && (
        <EmptyPlaceholder message={formatMessage(messages.noUsers)}>
          <UserAddButton onSave={handleSave} />
        </EmptyPlaceholder>
      )}
    </Page>
  )
}

export default UsersList