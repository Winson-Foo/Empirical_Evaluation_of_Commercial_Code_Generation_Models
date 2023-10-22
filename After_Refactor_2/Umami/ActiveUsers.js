import { useMemo } from 'react';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';
import styles from './ActiveUsers.module.css';

// Extract the hardcoded string into a constant for better maintainability.
const ACTIVE_USERS_QUERY_KEY = 'websites:active';

function ActiveUsers({ websiteId, value, refetchInterval = 60000 }) {
  // Destructure the variables needed from useMessages and useApi hooks.
  const { formatMessage, messages } = useMessages();
  const { get, useQuery } = useApi();

  // Use the extracted constant for the query key.
  const { data } = useQuery(
    [ACTIVE_USERS_QUERY_KEY, websiteId],
    () => get(`/websites/${websiteId}/active`),
    {
      refetchInterval,
      // Simplify the enabled condition.
      enabled: Boolean(websiteId),
    },
  );

  // Use a descriptive variable name and simplify the count calculation.
  const activeUsersCount = useMemo(() => {
    return websiteId ? data?.[0]?.x || 0 : value ?? 0;
  }, [data, value, websiteId]);

  // Use an early return instead of an if statement.
  if (!activeUsersCount) return null;

  return (
    // Use a more generic component instead of tying the logic to a specific one.
    <div className={styles.activeUsers}>
      <span className={styles.activeUsersCount}>
        {formatMessage(messages.activeUsers, { x: activeUsersCount })}
      </span>
    </div>
  );
}

export default ActiveUsers;