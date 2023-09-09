import React from 'react';
import { Menu, Icon, Text, PopupTrigger, Popup, Item, Button } from 'react-basics';
import Icons from 'components/icons';
import { saveDashboard } from 'store/dashboard';
import useMessages from 'hooks/useMessages';

// Array of options for the menu
const menuOptions = [
  {
    label: 'Toggle Charts',
    value: 'charts',
  },
  {
    label: 'Edit Dashboard',
    value: 'order',
  },
];

// Function to handle menu item selection
function handleMenuSelect(value) {
  if (value === 'charts') {
    saveDashboard(state => ({ showCharts: !state.showCharts }));
  }
  if (value === 'order') {
    saveDashboard({ editing: true });
  }
}

// Component to render dashboard settings popup menu
function DashboardSettingsPopup() {
  const { formatMessage, labels } = useMessages();

  return (
    <PopupTrigger>
      <Button>
        <Icon>
          <Icons.Edit />
        </Icon>
        <Text>{formatMessage(labels.edit)}</Text>
      </Button>
      <Popup alignment="end">
        <Menu variant="popup" items={menuOptions} onSelect={handleMenuSelect}>
          {({ label, value }) => <Item key={value}>{formatMessage(label)}</Item>}
        </Menu>
      </Popup>
    </PopupTrigger>
  );
}

export default DashboardSettingsPopup;

