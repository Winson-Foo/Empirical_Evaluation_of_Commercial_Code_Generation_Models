import { useMemo } from 'react';
import { firstBy } from 'thenby';
import WebsiteChart from 'components/metrics/WebsiteChart';
import useDashboard from 'store/dashboard';
import styles from './WebsiteList.module.css';

export default function WebsiteChartList({ websites, showCharts, limit }) {
  const { websiteOrder } = useDashboard();

  // Order websites by websiteOrder, limiting to `limit` results
  const orderedWebsites = useMemo(() => orderWebsites(websites, websiteOrder, limit), [
    websites,
    websiteOrder,
    limit,
  ]);

  // Render each website on the page
  return (
    <div>
      {orderedWebsites.map(({ id, name, domain }) => (
        <div key={id} className={styles.website}>
          <WebsiteChart
            websiteId={id}
            name={name}
            domain={domain}
            showChart={showCharts}
            showDetailsButton={true}
          />
        </div>
      ))}
    </div>
  );
}

function orderWebsites(websites, websiteOrder, limit) {
  // Map each website to include an `order` property, indicating the position in `websiteOrder`
  const websitesWithOrder = websites.map(website => ({
    ...website,
    order: websiteOrder.indexOf(website.id) || 0,
  }));

  // Sort websites by `order`, then limit to `limit` results
  const orderedWebsites = websitesWithOrder.sort(firstBy('order')).slice(0, limit);

  return orderedWebsites;
}