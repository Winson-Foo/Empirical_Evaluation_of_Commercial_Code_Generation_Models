import { Button, Icon, Icons, Text } from 'react-basics';
import styles from './WebsiteTags.module.css';

function createWebsiteTags(items, websites, onClick) {
  return websites.map((websiteId) => {
    const website = items.find((item) => item.id === websiteId);
    return (
      <div key={websiteId} className={styles.tag}>
        <Button onClick={() => onClick(websiteId)} variant="primary" size="sm">
          <Text>
            <b>{`${website.name}`}</b>
          </Text>
          <<Icon><Icons.Close /></Icon>
        </Button>
      </div>
    );
  });
}

function WebsiteTags({ items, websites, onClick }) {
  if (websites.length === 0) {
    return null;
  }

  const tags = createWebsiteTags(items, websites, onClick);

  return <div className={styles.filters}>{tags}</div>;
}

export default WebsiteTags;

