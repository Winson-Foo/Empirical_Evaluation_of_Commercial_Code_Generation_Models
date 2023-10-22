import create from 'zustand';
import { getItem } from 'next-basics';

const localStorageKeys = {
  LOCALE: 'locale',
  THEME: 'theme',
  DATE_RANGE: 'dateRange',
};

export const constants = {
  DEFAULT_LOCALE: 'en-US',
  DEFAULT_THEME: 'light',
  DEFAULT_DATE_RANGE: {
    startDate: null,
    endDate: null,
  },
};

const getLocalStorageValue = key => getItem(key) || constants[key];

const initialState = {
  locale: getLocalStorageValue(localStorageKeys.LOCALE),
  theme: getLocalStorageValue(localStorageKeys.THEME),
  dateRange: getLocalStorageValue(localStorageKeys.DATE_RANGE),
  shareToken: null,
  user: null,
  config: null,
};

const store = create(() => ({ ...initialState }));

export function setState(newState) {
  store.setState({ ...store.getState(), ...newState });
}

export default store;