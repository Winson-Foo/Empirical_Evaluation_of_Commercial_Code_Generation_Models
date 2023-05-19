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

// hooks/useApi.js
import Axios from "axios";

export default function useApi() {
  function get(url) {
    return Axios.get(url).then(response => response.data);
  }

  function post(url, data) {
    return Axios.post(url, data).then(response => response.data);
  }

  function put(url, data) {
    return Axios.put(url, data).then(response => response.data);
  }

  function remove(url) {
    return Axios.delete(url).then(response => response.data);
  }

  function useQuery(key, fn, config = {}) {
    return useQuery(key, () => fn(), config);
  }

  return { get, post, put, remove, useQuery };
}

// hooks/useMessages.js
import { useState } from "react";
import enUsMessages from "../locales/en-US.json";

export default function useMessages() {
  const [locale] = useState("en-US"); // change as per the user's language
  const messages = { "en-US": enUsMessages };

  const formatMessage = messageKey => messages[locale][messageKey] || messageKey;
  const labels = messages[locale];

  return { formatMessage, labels };
}

// locales/en-US.json
{
  "selectWebsite": "Select a Website",
  "websiteName": "Website Name",
}

// App.js
import WebsiteSelect from './components/websiteSelect';

function App() {
  function handleWebsiteSelect(websiteId) {
    console.log("Selected Website ID: ", websiteId);
  }

  return (
    <div>
      <WebsiteSelect websiteId={1} onSelect={handleWebsiteSelect} />
    </div>
  );
}

export default App;

