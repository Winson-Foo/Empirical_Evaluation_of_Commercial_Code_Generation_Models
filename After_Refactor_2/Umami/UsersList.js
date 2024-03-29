import React from 'react';
import useApi from 'hooks/useApi';
import useUser from 'hooks/useUser';
import { useToast } from 'react-basics';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import EmptyPlaceholder from 'components/common/EmptyPlaceholder';
import UsersTable from './UsersTable';
import UserAddButton from './UserAddButton';
import useMessages from 'hooks/useMessages';

const UsersList = () => {
  const { formatMessage, labels, messages } = useMessages();
  const { user } = useUser();
  const { get, useQuery } = useApi();
  const { data, isLoading, error, refetch } = useQuery(['user'], () => get(`/users`), {
    enabled: !!user,
  });
  const { toast, showToast } = useToast();
  const hasData = data && data.length !== 0;

  const handleSave = async () => {
    await refetch();
    showToast({ message: formatMessage(messages.saved), variant: 'success' });
  };

  const handleDelete = async () => {
    await refetch();
    showToast({ message: formatMessage(messages.userDeleted), variant: 'success' });
  };

  const renderEmptyPlaceholder = () => {
    return (
      <EmptyPlaceholder message={formatMessage(messages.noUsers)}>
        <UserAddButton onSave={handleSave} />
      </EmptyPlaceholder>
    );
  };

  const renderUsersTable = () => {
    return <UsersTable data={data} onDelete={handleDelete} />;
  };

  return (
    <Page loading={isLoading} error={error}>
      {toast}
      <PageHeader title={formatMessage(labels.users)}>
        <UserAddButton onSave={handleSave} />
      </PageHeader>
      {hasData ? renderUsersTable() : renderEmptyPlaceholder()}
    </Page>
  );
};

export default UsersList;

