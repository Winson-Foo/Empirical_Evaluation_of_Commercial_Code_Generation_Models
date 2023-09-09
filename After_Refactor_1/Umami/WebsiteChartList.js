import { useMemo } from 'react';
import { firstBy } from 'thenby';
import WebsiteChart from 'components/metrics/WebsiteChart';
import useDashboard from 'store/dashboard';
import styles from './WebsiteList.module.css';

const DEFAULT_LIMIT = 10;

export default function WebsiteChartList({ websites = [], showCharts = false, limit = DEFAULT_LIMIT }) {
  const { websiteOrder = [] } = useDashboard();

  const orderedWebsites = useMemo(() => {
    return websites
      .map(website => {
        const orderIndex = websiteOrder.indexOf(website.id) || 0;
        return {...website, order: orderIndex};
      })
      .sort(firstBy('order'));
  }, [websites, websiteOrder]);

  function renderWebsite(website) {
    const {id, name, domain} = website;
    return (
      <div key={id} className={styles.website}>
        <WebsiteChart
          websiteId={id}
          name={name}
          domain={domain}
          showChart={showCharts}
          showDetailsButton={true}
        />
      </div>
    )
  }

  function renderWebsiteList() {
    return (
      <div>
        {orderedWebsites.slice(0, limit).map(renderWebsite)}
      </div>
    )
  }

  return renderWebsiteList();
}