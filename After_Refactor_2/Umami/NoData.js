import classNames from 'classnames';
import styles from './NoData.module.css';
import useMessages from 'hooks/useMessages';

const NoData = ({ className }) => {
  const { formatMessage, messages } = useMessages();

  return (
    <div className={classNames(styles.container, className)}>
      {formatMessage(messages.noDataAvailable)}
    </div>
  );
}

export default NoData;

// To improve maintainability, I've added a const function declaration for NoData with an arrow function and removed the named function declaration.

