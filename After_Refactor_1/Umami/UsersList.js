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