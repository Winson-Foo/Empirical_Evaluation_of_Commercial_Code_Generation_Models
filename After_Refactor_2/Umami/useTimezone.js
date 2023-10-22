import { useState, useCallback } from 'react';
import { getTimezone } from 'lib/date';
import { getItem, setItem } from 'next-basics';
import { TIMEZONE_CONFIG } from 'lib/constants';

const INITIAL_TIMEZONE = getItem(TIMEZONE_CONFIG) || getTimezone();

const saveTimezoneToStorage = (value) => {
  setItem(TIMEZONE_CONFIG, value);
};

const useTimezone = () => {
  const [timezone, setTimezone] = useState(INITIAL_TIMEZONE);

  const saveTimezone = useCallback((value) => {
    saveTimezoneToStorage(value);
    setTimezone(value);
  }, [setTimezone]);

  return [timezone, saveTimezone];
};

export default useTimezone;

