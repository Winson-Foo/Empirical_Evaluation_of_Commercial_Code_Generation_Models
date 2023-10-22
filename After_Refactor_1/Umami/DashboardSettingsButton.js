import React, { useCallback } from 'react';
import { Menu, Icon, Text, PopupTrigger, Popup, Item, Button } from 'react-basics';
import Icons from 'components/icons';
import { saveDashboard } from 'store/dashboard';
import useMessages from 'hooks/useMessages';

function handleSelect(value) {
  if (value === 'charts') {
    saveDashboard(state => ({ showCharts: !state.showCharts }));
  }
  if (value === 'order') {
    saveDashboard({ editing: true });
  }
}

function DashboardSettings() {
  const { formatMessage, labels } = useMessages();

  const menuOptions = [
    {
      label: formatMessage(labels.toggleCharts),
      value: 'charts',
    },
    {
      label: formatMessage(labels.editDashboard),
      value: 'order',
    },
  ];

  const renderMenuItem = useCallback(({ label, value }) => (
    <Item key={value}>{label}</Item>
  ), []);

  return (
    <PopupTrigger>
      <Button>
        <Icon>
          <Icons.Edit />
        </Icon>
        <Text>{formatMessage(labels.edit)}</Text>
      </Button>
      <Popup alignment="end">
        <Menu variant="popup" items={menuOptions} onSelect={handleSelect}>
          {renderMenuItem}
        </Menu>
      </Popup>
    </PopupTrigger>
  );
}

export default DashboardSettings;

