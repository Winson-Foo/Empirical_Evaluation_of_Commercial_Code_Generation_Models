import React from 'react';
import { Button, Icon, Icons, Text } from 'react-basics';
import styles from './WebsiteTags.module.css';

function WebsiteTags({ items = [], websites = [], onClick = () => {} }) {
  if (!websites.length) {
    return null;
  }

  const renderWebsite = (websiteId) => {
    const website = items.find(({ id }) => id === websiteId);

    if (!website) {
      return null;
    }

    const { name } = website;

    return (
      <div key={websiteId} className={styles.tag}>
        <Button onClick={() => onClick(websiteId)} variant="primary" size="sm">
          <Text>
            <b>{`${name}`}</b>
          </Text>
          <Icon>
            <Icons.Close />
          </Icon>
        </Button>
      </div>
    );
  };

  return <div className={styles.filters}>{websites.map(renderWebsite)}</div>;
}

export default WebsiteTags;

