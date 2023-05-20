import { useState } from 'react';
import { Button, Icon, Item, Menu, Popup, PopupTrigger } from 'react-basics';
import { saveDashboard } from 'store/dashboard';
import { Edit } from 'components/icons';
import useMessages from 'hooks/useMessages';

const SettingsMenu = ({ options, onSelect }) => (
  <Menu variant="popup" items={options} onSelect={onSelect}>
    {({ label, value }) => (
      <Item key={value}>
        {label}
      </Item>
    )}
  </Menu>
);

const DashboardSettingsButton = () => {
  const { formatMessage, labels } = useMessages();
  const [showCharts, setShowCharts] = useState(false);

  const handleSelect = (value) => {
    if (value === 'charts') {
      setShowCharts(!showCharts);
      saveDashboard({ showCharts: !showCharts });
    }
    if (value === 'order') {
      saveDashboard({ editing: true });
    }
  };

  const menuOptions = [
    { label: formatMessage(labels.toggleCharts), value: 'charts' },
    { label: formatMessage(labels.editDashboard), value: 'order' },
  ];

  return (
    <PopupTrigger>
      <Button>
        <Icon>
          <Edit />
        </Icon>
        <span>{formatMessage(labels.edit)}</span>
      </Button>
      <Popup alignment="end">
        <SettingsMenu options={menuOptions} onSelect={handleSelect} />
      </Popup>
    </PopupTrigger>
  );
};

export default DashboardSettingsButton;