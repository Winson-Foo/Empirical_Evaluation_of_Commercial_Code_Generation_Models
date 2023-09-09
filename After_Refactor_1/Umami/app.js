// constants.js
export const DATE_RANGE_CONFIG = 'dateRange';
export const DEFAULT_DATE_RANGE = { start: null, end: null };
export const DEFAULT_LOCALE = 'en-US';
export const DEFAULT_THEME = 'light';
export const LOCALE_CONFIG = 'locale';
export const THEME_CONFIG = 'theme';

// state.js
import create from 'zustand';
import { getItem } from 'next-basics';
import {
  DATE_RANGE_CONFIG,
  DEFAULT_DATE_RANGE,
  DEFAULT_LOCALE,
  DEFAULT_THEME,
  LOCALE_CONFIG,
  THEME_CONFIG,
} from './constants';

const initialState = {
  locale: getItem(LOCALE_CONFIG) || DEFAULT_LOCALE,
  theme: getItem(THEME_CONFIG) || DEFAULT_THEME,
  dateRange: getItem(DATE_RANGE_CONFIG) || DEFAULT_DATE_RANGE,
  shareToken: null,
  user: null,
  config: null,
};

const store = create(() => ({ ...initialState }));

export default store;

// actions.js
import store from './state';

export function setTheme(theme) {
  store.setState({ theme });
}

export function setLocale(locale) {
  store.setState({ locale });
}

export function setShareToken(shareToken) {
  store.setState({ shareToken });
}

export function setUser(user) {
  store.setState({ user });
}

export function setConfig(config) {
  store.setState({ config });
}

export function setDateRange(dateRange) {
  store.setState({ dateRange });
}