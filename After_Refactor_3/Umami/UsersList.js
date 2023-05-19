import React, { useState, useEffect } from 'react';
import { useToast } from 'react-basics';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import EmptyPlaceholder from 'components/common/EmptyPlaceholder';
import UsersTable from './UsersTable';
import UserAddButton from './UserAddButton';
import useApi from 'hooks/useApi';
import useUser from 'hooks/useUser';
import useMessages from 'hooks/useMessages';

const USERS_QUERY_KEY = 'users';

const UsersList = () => {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const { formatMessage, labels, messages } = useMessages();
  const { user } = useUser();
  const { get } = useApi();
  const { toast, showToast } = useToast();

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const result = await get(`/users`);
        setUsers(result);
      } catch (err) {
        setError(err.message || 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    if (user) {
      fetchUsers();
    }
  }, [get, user]);

  const handleSave = async () => {
    try {
      await fetchUsers();
      showToast({ message: formatMessage(messages.saved), variant: 'success' });
    } catch (err) {
      showError(err);
    }
  };

  const handleDelete = async () => {
    try {
      await fetchUsers();
      showToast({ message: formatMessage(messages.userDeleted), variant: 'success' });
    } catch (err) {
      showError(err);
    }
  };

  const showError = (err) => {
    setError(err.message || 'Unknown error');
    showToast({ message: formatMessage(messages.error), variant: 'error' });
  };

  const hasUsers = users.length !== 0;

  return (
    <Page loading={isLoading} error={error}>
      {toast}
      <PageHeader title={formatMessage(labels.users)}>
        <UserAddButton onSave={handleSave} />
      </PageHeader>
      {hasUsers && <UsersTable data={users} onDelete={handleDelete} />}
      {!hasUsers && (
        <EmptyPlaceholder message={formatMessage(messages.noUsers)}>
          <UserAddButton onSave={handleSave} />
        </EmptyPlaceholder>
      )}
    </Page>
  );
};

export default UsersList;