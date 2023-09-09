import React from 'react';
import PropTypes from 'prop-types';
import { Column, Row, Text } from 'react-basics';
import ActiveUsers from './ActiveUsers';
import Favicon from 'components/common/Favicon';
import styles from './WebsiteHeader.module.css';

const WebsiteHeader = ({ websiteId, websiteName, websiteDomain, children }) => {
  return (
    <Row className={styles.header} justifyContent="center">
      <Column className={styles.titleWrapper} variant="two">
        <Favicon domain={websiteDomain} />
        <Text>{websiteName}</Text>
      </Column>
      <Column className={styles.infoWrapper} variant="two">
        <ActiveUsers websiteId={websiteId} />
        {children}
      </Column>
    </Row>
  );
};

WebsiteHeader.propTypes = {
  websiteId: PropTypes.string.isRequired,
  websiteName: PropTypes.string.isRequired,
  websiteDomain: PropTypes.string.isRequired,
  children: PropTypes.node,
};

export default WebsiteHeader;

