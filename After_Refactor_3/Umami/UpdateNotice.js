import { useState, useEffect, useCallback } from 'react';
import { Button, Row, Column } from 'react-basics';
import { setItem } from 'next-basics';
import { checkVersion } from 'store/version';
import { REPO_URL, VERSION_CHECK } from 'lib/constants';
import styles from './UpdateNotice.module.css';
import useMessages from 'hooks/useMessages';

function UpdateMessage({ latest, formatMessage, messages }) {
  return (
    <Column variant="two" className={styles.message}>
      {formatMessage(messages.newVersionAvailable, { version: `v${latest}` })}
    </Column>
  );
}

function DismissButton({ formatMessage, labels, onClick }) {
  return (
    <Button onClick={onClick}>{formatMessage(labels.dismiss)}</Button>
  );
}

function UpdateButton({ formatMessage, labels, onClick }) {
  return (
    <Button variant="primary" onClick={onClick}>
      {formatMessage(labels.viewDetails)}
    </Button>
  );
}

function handleOpenReleaseUrl(releaseUrl) {
  return () => open(releaseUrl || REPO_URL, '_blank');
}

function useUpdateCheckEffect(latest) {
  const updateCheck = useCallback(() => {
    setItem(VERSION_CHECK, { version: latest, time: Date.now() });
  }, [latest]);

  useEffect(() => {
    updateCheck();
  }, [latest, updateCheck]);
}

export function UpdateNotice() {
  const { formatMessage, labels, messages } = useMessages();
  const { latest, checked, hasUpdate, releaseUrl } = useStore();
  const [dismissed, setDismissed] = useState(false);

  useUpdateCheckEffect(latest);

  function handleDismissClick() {
    setDismissed(true);
  }

  function handleViewClick() {
    handleOpenReleaseUrl(releaseUrl)();
    setDismissed(true);
  }

  useEffect(() => {
    if (!checked) {
      checkVersion();
    }
  }, [checked]);

  if (!hasUpdate || dismissed) {
    return null;
  }

  return (
    <Row className={styles.notice}>
      <UpdateMessage 
        latest={latest} 
        formatMessage={formatMessage} 
        messages={messages} 
      />
      <Column className={styles.buttons}>
        <UpdateButton 
          formatMessage={formatMessage} 
          labels={labels} 
          onClick={handleViewClick} 
        />
        <DismissButton 
          formatMessage={formatMessage} 
          labels={labels} 
          onClick={handleDismissClick} 
        />
      </Column>
    </Row>
  );
}

export default UpdateNotice;