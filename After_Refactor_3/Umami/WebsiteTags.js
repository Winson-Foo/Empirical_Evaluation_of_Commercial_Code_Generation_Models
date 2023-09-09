import { Button, Icon, Icons, Text } from 'react-basics';
import styles from './WebsiteTags.module.css';

/**
 * Renders a list of website tags.
 *
 * @param {Array} items - List of website items.
 * @param {Array} websites - List of selected website ids.
 * @param {Function} onClick - Function to call when a tag is clicked.
 */
export function WebsiteTags({ items=[], websites=[], onClick }) {

  // If there are no selected websites, render nothing
  if (websites.length === 0) {
    return null;
  }

  return (
    <div className={styles.filters}>
      {websites.map(websiteId => {

        // Find the corresponding website item
        const foundWebsite = items.find(item => item.id === websiteId);

        return (
          <div key={websiteId} className={styles.tag}>

            {/* Render each tag */}
            <Button onClick={() => onClick(websiteId)} variant="primary" size="sm">
              <Text>
                <b>{`${foundWebsite.name}`}</b>
              </Text>
              <Icon>
                <Icons.Close />
              </Icon>
            </Button>

          </div>
        );
      })}
    </div>
  );
}

export default WebsiteTags;