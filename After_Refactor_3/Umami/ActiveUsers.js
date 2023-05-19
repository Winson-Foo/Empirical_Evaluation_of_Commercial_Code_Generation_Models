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