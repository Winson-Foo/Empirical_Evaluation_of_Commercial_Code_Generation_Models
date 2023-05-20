import React from 'react';
import PropTypes from 'prop-types';
import { Icon, Icons } from 'react-basics';
import classNames from 'classnames';
import Link from 'next/link';
import { safeDecodeURI } from 'next-basics';
import usePageQuery from 'hooks/usePageQuery';
import useMessages from 'hooks/useMessages';
import styles from './FilterLink.module.css';

const FilterLink = ({
  id,
  value,
  label,
  externalUrl,
  children,
  className,
}) => {
  const { formatMessage, labels } = useMessages();
  const { resolveUrl, query } = usePageQuery();

  const isActive = query[id] !== undefined;
  const isSelected = query[id] === value;

  const getLinkLabel = () =>
    value ? safeDecodeURI(label || value) : `(${label || formatMessage(labels.unknown)})`;

  const renderLink = () => (
    <Link href={resolveUrl({ [id]: value })} className={styles.label} replace>
      {getLinkLabel()}
    </Link>
  );

  const renderExternalLink = () => (
    <a className={styles.link} href={externalUrl} target="_blank" rel="noreferrer noopener">
      <Icon className={styles.icon}>
        <Icons.External />
      </Icon>
    </a>
  );

  const getClassName = () =>
    classNames(styles.row, className, {
      [styles.inactive]: isActive && !isSelected,
      [styles.active]: isActive && isSelected,
    });

  return <div className={getClassName()}>{children}{externalUrl ? renderExternalLink() : renderLink()}</div>;
};

FilterLink.propTypes = {
  id: PropTypes.string.isRequired,
  value: PropTypes.string,
  label: PropTypes.string,
  externalUrl: PropTypes.string,
  children: PropTypes.node,
  className: PropTypes.string,
};
FilterLink.defaultProps = {
  value: '',
  label: '',
  externalUrl: '',
  children: null,
  className: '',
};

export default FilterLink;


