import { useMemo } from 'react';
import { StatusLight } from 'react-basics';
import useActiveUsers from 'hooks/useActiveUsers';
import useMessages from 'hooks/useMessages';
import styles from './ActiveUsers.module.css';

export function ActiveUsers({ websiteId, value, refetchInterval = 60000 }) {
  const { formatMessage, messages } = useMessages();
  const { count } = useActiveUsers(websiteId, value, refetchInterval);

  if (count === 0) {
    return null;
  }

  return (
    <StatusLight variant="success">
      <div className={styles.text}>
        {formatMessage(messages.activeUsers, { count })}
      </div>
    </StatusLight>
  );
}

export default ActiveUsers;


// useActiveUsers hook

import { useMemo } from 'react';
import useApi from './useApi';

function useActiveUsers(websiteId, value, refetchInterval) {
  const { get, useQuery } = useApi();

  const { data } = useQuery(
    ['websites:active', websiteId],
    () => get(`/websites/${websiteId}/active`),
    {
      refetchInterval,
      enabled: !!websiteId,
    },
  );

  const count = useMemo(() => calculateCount(data, value, websiteId), [
    data,
    value,
    websiteId,
  ]);

  return { count };
}

function calculateCount(data, value, websiteId) {
  if (websiteId) {
    return data?.[0]?.x || 0;
  }

  return value !== undefined ? value : 0;
}

export default useActiveUsers;

