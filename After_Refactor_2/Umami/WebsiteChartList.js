import { useMemo } from 'react';
import { firstBy } from 'thenby';
import WebsiteChart from 'components/metrics/WebsiteChart';
import useDashboard from 'store/dashboard';
import styles from './WebsiteList.module.css';

export default function WebsiteChartList({ websites, showCharts, limit }) {
  const { websiteOrder } = useDashboard();

  const sortedWebsites = useMemo(() => sortWebsites(websites, websiteOrder), [
    websites,
    websiteOrder,
  ]);

  return <div>{renderWebsiteCharts(sortedWebsites, showCharts, limit)}</div>;
}

function sortWebsites(websites, websiteOrder) {
  return websites
    .map(website => ({
      id: website.id,
      name: website.name,
      domain: website.domain,
      order: websiteOrder.indexOf(website.id) || 0,
    }))
    .sort(firstBy('order'));
}

function renderWebsiteCharts(websites, showCharts, limit) {
  return websites.slice(0, limit).map(({ id, name, domain }) => (
    <div key={id} className={styles.website}>
      <WebsiteChart
        websiteId={id}
        name={name}
        domain={domain}
        showChart={showCharts}
        showDetailsButton={true}
      />
    </div>
  ));
}

