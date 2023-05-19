import React from 'react';
import classNames from 'classnames';
import { Banner, Loading } from 'react-basics';
import useMessages from 'hooks/useMessages';
import styles from './LoadingPage.module.css';

function ErrorBanner({ message }) {
  const { formatMessage } = useMessages();
  return <Banner variant="error">{formatMessage(message)}</Banner>;
}

function LoadingSpinner({ position }) {
  return <Loading icon="spinner" size="xl" position={position} />;
}

function LoadingPage({ className, error, loading, children }) {
  return (
    <>
      {error && <ErrorBanner message={error} />}
      {loading && <LoadingSpinner position="page" />}
      {!error && !loading && (
        <div className={classNames(styles.page, className)}>
          {children}
        </div>
      )}
    </>
  );
}

export default LoadingPage;

