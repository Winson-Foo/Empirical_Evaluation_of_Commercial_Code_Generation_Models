// useCountryNames.js
import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/router';
import { get } from 'next-basics';
import countryNames from './countryNames';

export default function useCountryNames(locale) {
  const [list, setList] = useState(countryNames[locale] || enUS);
  const { basePath } = useRouter();

  const fetchData = async (locale) => {
    try {
      const { data } = await get(`${basePath}/intl/country/${locale}.json`);

      if (data) {
        countryNames[locale] = data;
        setList(countryNames[locale]);
      } else {
        setList(enUS);
      }
    } catch (error) {
      setList(enUS);
    }
  }

  useEffect(() => {
    if (!countryNames[locale]) {
      fetchData(locale);
    } else {
      setList(countryNames[locale]);
    }
  }, [locale]);

  const memoizedList = useMemo(() => list, [list]);

  return memoizedList;
}

