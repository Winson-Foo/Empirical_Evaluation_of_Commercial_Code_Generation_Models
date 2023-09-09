import create from 'zustand';
import { getItem } from 'next-basics';
import {
  DATE_RANGE_CONFIG,
  DEFAULT_DATE_RANGE,
  DEFAULT_LOCALE,
  DEFAULT_THEME,
  LOCALE_CONFIG,
  THEME_CONFIG,
} from 'lib/constants';

const initialState = {
  config: null,
  dateRange: getItem(DATE_RANGE_CONFIG) || DEFAULT_DATE_RANGE,
  locale: getItem(LOCALE_CONFIG) || DEFAULT_LOCALE,
  shareToken: null,
  theme: getItem(THEME_CONFIG) || DEFAULT_THEME,
  user: null,
};

function createStore() {
  return create(() => initialState);
}

export const setTheme = (theme) => store.setState({ theme });
export const setLocale = (locale) => store.setState({ locale });
export const setShareToken = (shareToken) => store.setState({ shareToken });
export const setUser = (user) => store.setState({ user });
export const setConfig = (config) => store.setState({ config });
export const setDateRange = (dateRange) => store.setState({ dateRange });

export { initialState };
export default createStore();