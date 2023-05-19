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

// renderLink.test.js
import { renderLink } from "./renderLink";
import FilterLink from "components/common/FilterLink";

describe("renderLink", () => {
  it("should render a FilterLink with the correct props", () => {
    const referrer = "example.com";
    const expected = (
      <FilterLink
        id="referrer"
        value={referrer}
        externalUrl={`https://${referrer}`}
      />
    );
    const result = renderLink({ x: referrer });
    expect(result).toEqual(expected);
  });

  it("should render a FilterLink with the 'none' label when referrer is falsy", () => {
    const referrer = null;
    const expected = (
      <FilterLink id="referrer" value={referrer} label="None" />
    );
    const result = renderLink({ x: referrer });
    expect(result).toEqual(expected);
  });
});

// renderLink.js
import FilterLink from "components/common/FilterLink";

/**
 * Renders a FilterLink for the given referrer
 * @param {string} x - The referrer to render a link for
 * @param {string} label - Optional label to use instead of value for FilterLink
 * @returns {JSX.Element} - The rendered FilterLink component
 */
export function renderLink({ x: referrer, label }) {
  const value = referrer ?? null;
  const externalUrl = value ? `https://${value}` : null;
  const linkLabel = label || value || "None";

  return (
    <FilterLink
      id="referrer"
      value={value}
      externalUrl={externalUrl}
      label={linkLabel}
    />
  );
}

export default renderLink;

