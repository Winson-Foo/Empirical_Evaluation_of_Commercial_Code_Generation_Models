import { useMemo } from 'react';
import { StatusLight } from 'react-basics';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';
import PropTypes from 'prop-types';

import styles from './ActiveUsers.module.css';

function ActiveUsers({ websiteId, value, refetchInterval }) {
  const { formatMessage, messages } = useMessages();
  const { get, useQuery } = useApi();
  const { data } = useQuery(
    ['websites:active', websiteId],
    () => get(`/websites/${websiteId}/active`),
    {
      refetchInterval,
      enabled: !!websiteId,
    },
  );

  const count = useMemo(() => {
    if (websiteId) {
      return data?.[0]?.x || 0;
    }

    return value !== undefined ? value : 0;
  }, [data, value, websiteId]);

  if (count === 0) {
    return null;
  }

  return (
    <StatusLight variant="success">
      <div className={styles.text}>{formatMessage(messages.activeUsers, { x: count })}</div>
    </StatusLight>
  );
}

ActiveUsers.propTypes = {
  websiteId: PropTypes.number,
  value: PropTypes.number,
  refetchInterval: PropTypes.number,
};

ActiveUsers.defaultProps = {
  refetchInterval: 60000,
};

export default ActiveUsers;