// ReferrersTable.js
import MetricsTable from "./MetricsTable";
import FilterLink from "components/common/FilterLink";
import useMessages from "hooks/useMessages";

/**
 * Renders the ReferrersTable component with the given websiteId and other props
 * @param {string} websiteId - The id of the website
 * @param {object} props - Other props passed down to MetricsTable
 * @returns {JSX.Element} - The rendered ReferrersTable component
 */
export function ReferrersTable({ websiteId, ...props }) {
  const { formatMessage, labels } = useMessages();

  /**
   * Renders a FilterLink for the given referrer
   * @param {string} referrer - The referrer to render a link for
   * @returns {JSX.Element} - The rendered FilterLink component
   */
  const renderLink = ({ x: referrer }) => {
    return (
      <FilterLink
        id="referrer"
        value={referrer}
        externalUrl={`https://${referrer}`}
        label={!referrer && formatMessage(labels.none)}
      />
    );
  };

  return (
    <>
      <MetricsTable
        {...props}
        title={formatMessage(labels.referrers)}
        type="referrer"
        metric={formatMessage(labels.views)}
        websiteId={websiteId}
        renderLabel={renderLink}
      />
    </>
  );
}

export default ReferrersTable;