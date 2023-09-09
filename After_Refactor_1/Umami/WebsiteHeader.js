import React from 'react';
import { Row, Column, Text } from 'react-basics';
import Favicon from 'components/common/Favicon';
import ActiveUsers from './ActiveUsers';
import styles from './WebsiteHeader.module.css';

// Header component for a website
function WebsiteHeader({ id, name, children }) {
  // Render website name and favicon
  const renderTitle = () => (
    <Column className={styles.title} variant="two">
      <Favicon domain={name} />
      <Text>{name}</Text>
    </Column>
  );

  // Render active users and any child components
  const renderInfo = () => (
    <Column className={styles.info} variant="two">
      <ActiveUsers websiteId={id} />
      {children}
    </Column>
  );

  // Render header component
  return (
    <Row className={styles.header} justifyContent="center">
      {renderTitle()}
      {renderInfo()}
    </Row>
  );
}

export default WebsiteHeader;