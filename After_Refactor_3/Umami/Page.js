import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { Banner, Loading } from 'react-basics';
import useMessages from 'hooks/useMessages';
import styles from './Page.module.css';

function Page({ className = '', error = false, loading = false, children }) {
  const { formatMessage, messages } = useMessages();

  // Display an error banner if an error prop is provided
  if (error) {
    return <Banner variant="error">{formatMessage(messages.error)}</Banner>;
  }

  // Display a loading spinner if a loading prop is provided
  if (loading) {
    return <Loading icon="spinner" size="xl" position="page" />;
  }

  // Otherwise, render the children as a div with the specified class names
  return <div className={classNames(styles.page, className)}>{children}</div>;
}

Page.propTypes = {
  className: PropTypes.string,
  error: PropTypes.bool,
  loading: PropTypes.bool,
  children: PropTypes.node.isRequired,
};

export default Page;