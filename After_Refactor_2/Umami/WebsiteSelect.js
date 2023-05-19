import PropTypes from 'prop-types';
import { Dropdown, Item } from 'react-basics';
import useWebsiteApi from 'hooks/useWebsiteApi';
import useIntlMessages from 'hooks/useIntlMessages';

function WebsiteSelect({ websiteId, onSelect }) {
  const { formatMessage, labels } = useIntlMessages();
  const { data } = useWebsiteApi();

  const renderValue = value => {
    return data?.find(({ id }) => id === value)?.name;
  };

  return (
    <Dropdown
      items={data}
      value={websiteId}
      renderValue={renderValue}
      onChange={onSelect}
      alignment="end"
      placeholder={formatMessage(labels.selectWebsite)}
      style={{ width: 200 }}
    >
      {({ id, name }) => <Item key={id}>{name}</Item>}
    </Dropdown>
  );
}

WebsiteSelect.propTypes = {
  websiteId: PropTypes.number,
  onSelect: PropTypes.func.isRequired
};

export default WebsiteSelect;

And here's the updated `useWebsiteApi` hook:

import useApi from 'hooks/useApi';

function useWebsiteApi() {
  const { get, useQuery } = useApi();
  const { data } = useQuery(['websites:me'], () => get('/me/websites'));

  return { data };
}

export default useWebsiteApi;

