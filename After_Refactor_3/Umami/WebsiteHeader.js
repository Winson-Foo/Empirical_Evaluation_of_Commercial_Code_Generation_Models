import React from 'react';
import PropTypes from 'prop-types';
import { Row, Column } from 'react-basics';
import Favicon from 'components/common/Favicon';
import ActiveUsers from './ActiveUsers';
import styles from './WebsiteHeader.module.css';

function HeaderTitle({ name, domain }) {
  return (
    <Column className={styles.title} variant="two">
      <Favicon domain={domain} />
      <Text>{name}</Text>
    </Column>
  );
}

function CompanyInfo({ companyId, children }) {
  return (
    <Column className={styles.info} variant="two">
      <ActiveUsers companyId={companyId} />
      {children}
    </Column>
  );
}

export default function HeaderWithCompanyInfo({ companyId, name, domain, children }) {
  return (
    <Row className={styles.header} justifyContent="center">
      <HeaderTitle name={name} domain={domain} />
      <CompanyInfo companyId={companyId} children={children} />
    </Row>
  );
}

HeaderWithCompanyInfo.propTypes = {
  companyId: PropTypes.number.isRequired,
  name: PropTypes.string.isRequired,
  domain: PropTypes.string.isRequired,
  children: PropTypes.node,
};

