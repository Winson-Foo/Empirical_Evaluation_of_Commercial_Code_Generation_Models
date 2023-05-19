// websiteSelect.js
import React from "react";
import PropTypes from "prop-types";
import { Dropdown, Item } from 'react-basics';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';

function WebsiteSelect({ websiteId, onSelect }) {
  const { formatMessage, labels } = useMessages();
  const { get, useQuery } = useApi();

  const { data } = useQuery(["websites:me"], () => get("/me/websites"));

  const renderValue = value => (
    data?.find(({ id }) => id === value)?.name
  );

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
  websiteId: PropTypes.number.isRequired,
  onSelect: PropTypes.func.isRequired,
};

export default WebsiteSelect;