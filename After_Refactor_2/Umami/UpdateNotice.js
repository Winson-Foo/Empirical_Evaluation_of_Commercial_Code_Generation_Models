import { useState, useEffect, useCallback } from 'react';
import { Button, Row, Column } from 'react-basics';
import { setItem } from 'next-basics';
import useStore, { checkVersion } from 'store/version';
import { REPO_URL, VERSION_CHECK } from 'lib/constants';
import styles from './UpdateNotice.module.css';
import useMessages from 'hooks/useMessages';

function UpdateNoticeRow({ message, buttons }) {
  return <Row className={styles.notice}>{message}{buttons}</Row>;
}

function UpdateNoticeMessage({ message }) {
  return <Column variant="two" className={styles.message}>{message}</Column>;
}

function UpdateNoticeButtons({ viewButton, dismissButton }) {
  return <Column className={styles.buttons}>{viewButton}{dismissButton}</Column>;
}

export function UpdateNotice() {
  const { formatMessage, labels, messages } = useMessages();
  const { latestVersion, isChecked, hasUpdate, releaseUrl } = useStore();
  const [isDismissed, setIsDismissed] = useState(false);

  const handleViewDetailsClick = useCallback(() => {
    updateVersionCheck();
    setIsDismissed(true);
    open(releaseUrl || REPO_URL, '_blank');
  }, [releaseUrl]);

  const handleDismissClick = useCallback(() => {
    updateVersionCheck();
    setIsDismissed(true);
  }, []);

  const updateVersionCheck = useCallback(() => {
    setItem(VERSION_CHECK, { version: latestVersion, time: Date.now() });
  }, [latestVersion]);

  useEffect(() => {
    if (!isChecked) {
      checkVersion();
    }
  }, [isChecked]);

  if (!hasUpdate || isDismissed) {
    return null;
  }

  return (
    <UpdateNoticeRow
      message={<UpdateNoticeMessage
        message={formatMessage(messages.newVersionAvailable, { version: `v${latestVersion}` })}
      />}
      buttons={<UpdateNoticeButtons
        viewButton={<Button variant="primary" onClick={handleViewDetailsClick}>
          {formatMessage(labels.viewDetails)}
        </Button>}
        dismissButton={<Button onClick={handleDismissClick}>
          {formatMessage(labels.dismiss)}
        </Button>}
      />}
    />
  );
}

export default UpdateNotice;

