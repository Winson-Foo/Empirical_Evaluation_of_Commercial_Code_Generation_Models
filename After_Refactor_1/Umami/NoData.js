// formatMessage.js
import { useIntl } from 'react-intl';

export function formatMessage(id, values = {}) {
  const intl = useIntl();
  return intl.formatMessage({ id }, values);
}

// NoData.js
import React from 'react';
import { formatMessage } from './formatMessage';
import styles from './NoData.module.css';

export function NoData({ className }) {
  return (
    <div className={`${styles.noDataContainer} ${className}`}>
      {formatMessage('noDataAvailable')}
    </div>
  );
}

export default NoData;

// NoData.module.css
.noDataContainer {
  // add styles here
}

