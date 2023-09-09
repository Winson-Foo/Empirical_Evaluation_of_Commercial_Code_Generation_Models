import { useState, useEffect, useCallback } from 'react';

import { setItem } from 'next-basics';

import { Button, Row, Column } from 'react-basics';

import useStore, { checkVersion } from 'store/version';

import useMessages from 'hooks/useMessages';

import { REPO_URL, VERSION_CHECK } from 'lib/constants';

import styles from './UpdateNotice.module.css';

import { useState, useEffect, useCallback } from 'react';

import { setItem } from 'next-basics';

import { Button, Row, Column } from 'react-basics';

import useStore, { checkVersion } from 'store/version';

import useMessages from 'hooks/useMessages';

import { REPO_URL, VERSION_CHECK } from 'lib/constants';

import styles from './UpdateNotice.module.css';


export function UpdateNotice() {

  const { formatMessage, labels, messages } = useMessages();

  const { latestVersion, isChecked, hasNewUpdate, releaseUrl } = useStore();

  const [isDismissed, setIsDismissed] = useState(false);

  const saveLatestVersionCheck = useCallback(() => {
    setItem(VERSION_CHECK, { version: latestVersion, time: Date.now() });
  }, [latestVersion]);
  
  useEffect(() => {
    if (!isChecked) {
      checkVersion();
    }
  }, [isChecked]);
  
  function handleViewDetailsClick() {
    saveLatestVersionCheck();
    setIsDismissed(true);
    open(releaseUrl || REPO_URL, '_blank');
  }
  
  function handleDismissClick() {
    saveLatestVersionCheck();
    setIsDismissed(true);
  }
  
  if (!hasNewUpdate || isDismissed) {
    return null;
  }
  
  return (
    <Row className={styles.notice}>
      <Column variant="two" className={styles.message}>
        {formatMessage(messages.newVersionAvailable, { version: `v${latestVersion}` })}
      </Column>
      <Column className={styles.buttons}>
        <Button variant="primary" onClick={handleViewDetailsClick}>
          {formatMessage(labels.viewDetails)}
        </Button>
        <Button onClick={handleDismissClick}>{formatMessage(labels.dismiss)}</Button>
      </Column>
    </Row>
  );
}

export default UpdateNotice;