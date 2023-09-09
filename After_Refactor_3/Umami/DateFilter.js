import { useState } from 'react';
import { Icon, Modal, Dropdown, Item, Text, Flexbox } from 'react-basics';
import { endOfYear, isSameDay } from 'date-fns';

// Custom components
import CustomRange from './CustomRange';
import DatePickerForm from './DatePickerForm';

// Hooks
import useLocale from 'hooks/useLocale';
import useApi from 'hooks/useApi';
import useDateRange from 'hooks/useDateRange';
import useMessages from 'hooks/useMessages';

const options = [
  { label: 'Today', value: '1day' },
  { label: 'Last 24 hours', value: '24hour' },
  { label: 'Yesterday', value: '-1day' },
  { label: 'This week', value: '1week', divider: true },
  { label: 'Last 7 days', value: '7day' },
  { label: 'This month', value: '1month', divider: true },
  { label: 'Last 30 days', value: '30day' },
  { label: 'Last 90 days', value: '90day' },
  { label: 'This year', value: '1year' },
  { label: 'All time', value: 'all', divider: true },
  { label: 'Custom range', value: 'custom', divider: true },
];

function DateFilter({ websiteId, value, className }) {
  const { formatMessage, labels } = useMessages();
  const { get } = useApi();
  const [dateRange, setDateRange] = useDateRange(websiteId);
  const { startDate, endDate } = dateRange;
  const [showPicker, setShowPicker] = useState(false);
  const { locale } = useLocale();

  // Handle changes to date range
  async function handleDateChange(value) {
    if (value === 'all' && websiteId) {
      const data = await get(`/websites/${websiteId}`);
      if (data) {
        setDateRange({ value, ...getDateRangeValues(new Date(data.createdAt), Date.now()) });
      }
    } else if (value !== 'all') {
      setDateRange(value);
    }
  }

  // Render the selected date range value
  const renderValue = value => {
    return value === 'custom' ? (
      <CustomRange startDate={startDate} endDate={endDate} onClick={() => handleChange('custom')} />
    ) : (
      options.find(e => e.value === value).label
    );
  };

  // Handle changes to the selected date range
  const handleChange = value => {
    if (value === 'custom') {
      setShowPicker(true);
      return;
    }
    handleDateChange(value);
  };

  // Handle changes to the date range from the custom range picker
  const handlePickerChange = value => {
    setShowPicker(false);
    handleDateChange(value);
  };

  // Handle closing the custom range picker
  const handleClose = () => setShowPicker(false);

  return (
    <>
      <Dropdown
        className={className}
        items={options}
        renderValue={renderValue}
        value={value}
        alignment="end"
        onChange={handleChange}
      >
        {({ label, value, divider }) => (
          <Item key={value} divider={divider}>
            {label}
          </Item>
        )}
      </Dropdown>
      {showPicker && (
        <Modal onClose={handleClose}>
          <DatePickerForm
            startDate={startDate}
            endDate={endDate}
            minDate={new Date(2000, 0, 1)}
            maxDate={endOfYear(new Date())}
            onChange={handlePickerChange}
            onClose={() => setShowPicker(false)}
          />
        </Modal>
      )}
    </>
  );
}

export default DateFilter;

