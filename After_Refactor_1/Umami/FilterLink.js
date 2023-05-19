import React from 'react';
import PropTypes from 'prop-types';
import { Icons } from 'react-basics';
import classNames from 'classnames';
import Link from 'next/link';
import { safeDecodeURI } from 'next-basics';
import usePageQuery from 'hooks/usePageQuery';
import useMessages from 'hooks/useMessages';
import styles from './FilterLink.module.css';

function getLinkClassNames(active, selected, className) {
  return classNames(styles.row, className, {
    [styles.inactive]: active && !selected,
    [styles.active]: active && selected,
  });
}

function getLabel(label, formatMessage, labels, value) {
  if (!value) {
    return `(${label || formatMessage(labels.unknown)})`;
  }
  return safeDecodeURI(label || value);
}

function FilterLink({ id, value, label, externalUrl, children, className }) {
  const { formatMessage, labels } = useMessages();
  const { resolveUrl, query } = usePageQuery();
  const active = query[id] !== undefined;
  const selected = query[id] === value;
  const linkClassNames = getLinkClassNames(active, selected, className);
  const linkLabel = getLabel(label, formatMessage, labels, value);

  return (
    <div className={linkClassNames}>
      {children}
      {value && (
        <Link href={resolveUrl({ [id]: value })} className={styles.label} replace>
          {linkLabel}
        </Link>
      )}
      {!value && linkLabel}
      {externalUrl && (
        <a className={styles.link} href={externalUrl} target="_blank" rel="noreferrer noopener">
          <Icons.External className={styles.icon} />
        </a>
      )}
    </div>
  );
}

FilterLink.propTypes = {
  id: PropTypes.string.isRequired,
  value: PropTypes.string,
  label: PropTypes.string,
  externalUrl: PropTypes.string,
  children: PropTypes.node,
  className: PropTypes.string,
};

export default FilterLink;

