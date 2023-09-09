import { useState } from 'react';
import { Icon, Modal, Dropdown, Item, Text, Flexbox } from 'react-basics';
import { endOfYear, isSameDay } from 'date-fns';
import DatePickerForm from 'components/metrics/DatePickerForm';
import useLocale from 'hooks/useLocale';
import { dateFormat, getDateRangeValues } from 'lib/date';
import Icons from 'components/icons';
import useApi from 'hooks/useApi';
import useDateRange from 'hooks/useDateRange';
import useMessages from 'hooks/useMessages';

function DateFilter({ websiteId, value, className }) {
  const { formatMessage, labels } = useMessages();
  const { get } = useApi();
  const [dateRange, setDateRange] = useDateRange(websiteId);
  const { startDate, endDate } = dateRange;
  const [showPicker, setShowPicker] = useState(false);

  const options = getOptions();

  const renderValue = renderOptionValue(options, startDate, endDate);

  const handleOptionChange = optionValue => {
    if (optionValue === 'custom') {
      setShowPicker(true);
    } else {
      setDateRange(getDateRange(optionValue, websiteId, get));
    }
  };

  const handlePickerChange = pickerValue => {
    setShowPicker(false);
    setDateRange(getDateRange(pickerValue, websiteId, get));
  };

  const handleClose = () => setShowPicker(false);

  return (
    <>
      <Dropdown
        className={className}
        items={options}
        renderValue={renderValue}
        value={value}
        alignment="end"
        onChange={handleOptionChange}
      >
        {renderOptionItem}
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

function getOptions() {
  const { formatMessage, labels } = useMessages();
  const options = [
    { label: formatMessage(labels.today), value: '1day' },
    {
      label: formatMessage(labels.lastHours, { x: 24 }),
      value: '24hour',
    },
    {
      label: formatMessage(labels.yesterday),
      value: '-1day',
    },
    {
      label: formatMessage(labels.thisWeek),
      value: '1week',
      divider: true,
    },
    {
      label: formatMessage(labels.lastDays, { x: 7 }),
      value: '7day',
    },
    {
      label: formatMessage(labels.thisMonth),
      value: '1month',
      divider: true,
    },
    {
      label: formatMessage(labels.lastDays, { x: 30 }),
      value: '30day',
    },
    {
      label: formatMessage(labels.lastDays, { x: 90 }),
      value: '90day',
    },
    { label: formatMessage(labels.thisYear), value: '1year' },
    websiteId && {
      label: formatMessage(labels.allTime),
      value: 'all',
      divider: true,
    },
    {
      label: formatMessage(labels.customRange),
      value: 'custom',
      divider: true,
    },
  ].filter(option => option);

  return options;
}

function getDateRange(optionValue, websiteId, get) {
  if (optionValue === 'all' && websiteId) {
    const data = await get(`/websites/${websiteId}`);
    if (data) {
      return { value: optionValue, ...getDateRangeValues(new Date(data.createdAt), Date.now()) };
    }
  } else if (optionValue !== 'all') {
    return { value: optionValue };
  }
  return null;
}

function renderOptionValue(options, startDate, endDate) {
  const selectedOption = options.find(option => option.value === this.props.value);
  if (selectedOption && selectedOption.value === 'custom') {
    return (
      <CustomRange startDate={startDate} endDate={endDate} onClick={() => handleChange('custom')} />
    );
  }
  return selectedOption.label;
}

function renderOptionItem({ label, value, divider }) {
  return (
    <Item key={value} divider={divider}>
      {label}
    </Item>
  );
}

const CustomRange = ({ startDate, endDate, onClick }) => {
  const { locale } = useLocale();

  function handleClick(e) {
    e.stopPropagation();

    onClick();
  }

  return (
    <Flexbox gap={10} alignItems="center" wrap="nowrap">
      <Icon className="mr-2" onClick={handleClick}>
        <Icons.Calendar />
      </Icon>
      <Text>
        {dateFormat(startDate, 'd LLL y', locale)}
        {!isSameDay(startDate, endDate) && ` � ${dateFormat(endDate, 'd LLL y', locale)}`}
      </Text>
    </Flexbox>
  );
};

export default DateFilter;