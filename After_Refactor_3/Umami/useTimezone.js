import { useState, useCallback } from 'react';
import { getTimezone } from 'lib/date';
import { getItem, setItem } from 'next-basics';
import { TIMEZONE_CONFIG } from 'lib/constants';

// Get the timezone from local storage or use the default timezone.
const getDefaultTimezone = () => {
  return getItem(TIMEZONE_CONFIG) || getTimezone();
};

// Save the timezone to local storage and update the state.
const saveTimezone = (value, setTimezone) => {
  setItem(TIMEZONE_CONFIG, value);
  setTimezone(value);
};

// Custom hook for managing timezone.
const useTimezone = () => {
  const [timezone, setTimezone] = useState(getDefaultTimezone);

  // Memoize the saveTimezone function to improve performance.
  const memoizedSaveTimezone = useCallback((value) => {
    saveTimezone(value, setTimezone);
  }, [setTimezone]);

  return [timezone, memoizedSaveTimezone];
};

export default useTimezone;

