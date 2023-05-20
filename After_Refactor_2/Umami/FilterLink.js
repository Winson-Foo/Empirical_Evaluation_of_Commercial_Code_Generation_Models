import { Icon, Icons } from 'react-basics';
import classNames from 'classnames';
import Link from 'next/link';
import { safeDecodeURI } from 'next-basics';
import usePageQuery from 'hooks/usePageQuery';
import useMessages from 'hooks/useMessages';
import styles from './FilterLink.module.css';

function Label({ id, value, label, resolveUrl }) {
  const { formatMessage, labels } = useMessages();
  const selected = resolveUrl[id] === value;
  return (
    <>
      {!value && `(${label || formatMessage(labels.unknown)})`}
      {value && (
        <Link href={resolveUrl({ [id]: value })} className={styles.label} replace>
          {safeDecodeURI(label || value)}
        </Link>
      )}
    </>
  );
}

function ExternalLink({ externalUrl }) {
  return (
    <a className={styles.link} href={externalUrl} target="_blank" rel="noreferrer noopener">
      <Icon className={styles.icon}>
        <Icons.External />
      </Icon>
    </a>
  );
}

export function FilterLink({ id, value, label, externalUrl, children, className }) {
  const { resolveUrl, query } = usePageQuery();
  const active = query[id] !== undefined;

  return (
    <div
      className={classNames(styles.row, className, {
        [styles.inactive]: active && !selected,
        [styles.active]: active && selected,
      })}
    >
      {children}
      <Label id={id} value={value} label={label} resolveUrl={resolveUrl} />
      {externalUrl && <ExternalLink externalUrl={externalUrl} />}
    </div>
  );
}

export default FilterLink;