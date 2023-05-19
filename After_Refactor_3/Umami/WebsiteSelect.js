import { Dropdown, Item } from 'react-basics';
import { useApi } from 'hooks';
import { useMessages } from 'hooks';

type Website = {
  id: string;
  name: string;
};

type Props = {
  websiteId: string;
  onSelect: (value: string) => void;
};

export function WebsiteSelect({ websiteId, onSelect }: Props) {
  const { formatMessage, labels } = useMessages();
  const { data, isLoading, isError, refetch } = useWebsites();

  const renderWebsiteName = (website: Website): string => website.name;

  const handleSelect = (value: string): void => {
    onSelect(value);
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (isError) {
    return (
      <div>
        Error loading websites.{' '}
        <button onClick={refetch}>Retry</button>
      </div>
    );
  }

  return (
    <Dropdown
      items={data}
      value={websiteId}
      renderValue={renderWebsiteName}
      onChange={handleSelect}
      alignment="end"
      placeholder={formatMessage(labels.selectWebsite)}
      style={{ width: 200 }}
    >
      {(website: Website) => (
        <Item key={website.id}>{website.name}</Item>
      )}
    </Dropdown>
  );
}

function useWebsites() {
  const { get, useQuery } = useApi();
  return useQuery<Website[]>(['websites:me'], () => get('/me/websites'));
}

export default WebsiteSelect;