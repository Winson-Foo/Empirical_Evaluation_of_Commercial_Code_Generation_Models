import classNames from 'classnames';
import { Banner, Loading } from 'react-basics';
import { useMessages } from 'hooks';
import styles from './Page.module.css';

const Page = ({ className = '', error = false, loading = false, children }) => {
  const { formatMessage, messages } = useMessages();
  
  const errorBanner = error && <Banner variant="error">{formatMessage(messages.error)}</Banner>;
  const loadingIndicator = loading && <Loading icon="spinner" size="xl" position="page" />;
  
  return <div className={classNames(styles.page, className)}>{errorBanner || loadingIndicator || children}</div>;
}

export default Page;