import enUS from 'public/intl/country/en-US.json';

export const countryNames = {
  'en-US': enUS,
};

import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { get } from 'next-basics';
import { countryNames } from './countryNames';

/**
 * Hook that returns an object containing the country names for a given locale.
 * If the country names for the given locale are not available, it fetches them from the server.
 * @param {string} locale - The locale for which to get the country names.
 * @returns {Object} The object containing the country names for the given locale.
 */
export default function useCountryNames(locale) {
  const [list, setList] = useState(countryNames[locale] || enUS);
  const { basePath } = useRouter();

  /**
   * Fetches the country names for the given locale from the server and updates the countryNames object.
   * @param {string} locale - The locale for which to fetch the country names.
   */
  async function loadCountryNames(locale) {
    const { data } = await get(`${basePath}/intl/country/${locale}.json`);

    if (data) {
      countryNames[locale] = data;
      setList(countryNames[locale]);
    } else {
      setList(enUS);
    }
  }

  /**
   * Effect hook that loads the country names for the given locale when the component mounts or the locale changes.
   */
  useEffect(() => {
    if (!countryNames[locale]) {
      loadCountryNames(locale);
    } else {
      setList(countryNames[locale]);
    }
  }, [locale]);

  return list;
}