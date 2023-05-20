import { useState, useCallback } from 'react';
import { getTimezone } from 'lib/date';
import { getItem, setItem } from 'next-basics';
import { TIMEZONE_CONFIG } from 'lib/constants';

const TIMEZONE_STORAGE_KEY = TIMEZONE_CONFIG;

function useTimezone() {
  const getTimezoneFromStorageOrDefault = () => getItem(TIMEZONE_STORAGE_KEY) || getTimezone();
  const [timezone, setTimezone] = useState(getTimezoneFromStorageOrDefault());

  const saveTimezone = useCallback(
    (value) => {
      setItem(TIMEZONE_STORAGE_KEY, value);
      setTimezone(value);
    },
    [setTimezone],
  );
  
  return [timezone, saveTimezone];
}

export default useTimezone;

