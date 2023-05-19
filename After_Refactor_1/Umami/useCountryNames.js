import enUS from 'public/intl/country/en-US.json';

const countryNames = {
  'en-US': enUS,
};

export default countryNames;

import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { get } from 'next-basics';
import countryNames from './countryNames';

export default function useCountryNames(locale) {
  const [localizedCountryNames, setLocalizedCountryNames] = useState(countryNames[locale] || enUS);
  const { basePath } = useRouter();

  async function loadData(locale) {
    try {
      const { data } = await get(`${basePath}/intl/country/${locale}.json`);

      if (data) {
        countryNames[locale] = data;
        setLocalizedCountryNames(countryNames[locale]);
      }
    } catch (error) {
      console.error(`Failed to load localized country names for "${locale}"`, error);
    }
  }

  useEffect(() => {
    if (!countryNames[locale]) {
      loadData(locale);
    } else {
      setLocalizedCountryNames(countryNames[locale]);
    }
  }, [locale]);

  // Return localized country names
  return localizedCountryNames;
}