import moment from 'moment-timezone';
import { 
  parseISO, 
  addMinutes, 
  addHours, 
  addDays, 
  addMonths, 
  addYears, 
  subHours, 
  subDays, 
  subMonths, 
  subYears, 
  startOfMinute, 
  startOfHour, 
  startOfDay, 
  startOfWeek, 
  startOfMonth, 
  startOfYear, 
  endOfHour, 
  endOfDay, 
  endOfWeek, 
  endOfMonth, 
  endOfYear, 
  differenceInMinutes, 
  differenceInHours, 
  differenceInCalendarDays, 
  differenceInCalendarMonths, 
  differenceInCalendarYears, 
  format
} from 'date-fns';
import { getDateLocale } from 'lib/lang';

const DATE_FUNCS = {
  minute: [differenceInMinutes, addMinutes, startOfMinute],
  hour: [differenceInHours, addHours, startOfHour],
  day: [differenceInCalendarDays, addDays, startOfDay],
  month: [differenceInCalendarMonths, addMonths, startOfMonth],
  year: [differenceInCalendarYears, addYears, startOfYear],
};

/**
 * Returns the guessed timezone based on user's location
 * @return {string} - the timezone string
 */
export function getTimezone() {
  return moment.tz.guess();
}

/**
 * Converts a given datetime string to local time and returns it
 * @param {string} t - the datetime string to convert
 * @return {Date} - the converted date object in local timezone
 */
export function getLocalTime(t) {
  return addMinutes(new Date(t), new Date().getTimezoneOffset());
}

/**
 * Parses a date range value and returns an object with `startDate`, `endDate`, `unit`, and `value` properties
 * @param {string|object} value - the date range value to parse, can either be a string or an object with startDate and endDate properties
 * @param {string} locale - the locale string for the date format
 * @return {object} - the parsed date range object
 */
export function parseDateRange(value, locale = 'en-US') {
  if (typeof value === 'object') {
    const { startDate, endDate } = value;
    return {
      ...value,
      startDate: typeof startDate === 'string' ? parseISO(startDate) : startDate,
      endDate: typeof endDate === 'string' ? parseISO(endDate) : endDate,
    };
  }

  const now = new Date();
  const dateLocale = getDateLocale(locale);

  const match = value.match(/^(?<num>[0-9-]+)(?<unit>hour|day|week|month|year)$/);

  if (!match) return {};

  const { num, unit } = match.groups;

  if (+num === 1) {
    switch (unit) {
      case 'day':
        return {
          startDate: startOfDay(now),
          endDate: endOfDay(now),
          unit: 'hour',
          value,
        };
      case 'week':
        return {
          startDate: startOfWeek(now, { locale: dateLocale }),
          endDate: endOfWeek(now, { locale: dateLocale }),
          unit: 'day',
          value,
        };
      case 'month':
        return {
          startDate: startOfMonth(now),
          endDate: endOfMonth(now),
          unit: 'day',
          value,
        };
      case 'year':
        return {
          startDate: startOfYear(now),
          endDate: endOfYear(now),
          unit: 'month',
          value,
        };
    }
  }

  if (+num === -1) {
    switch (unit) {
      case 'day':
        return {
          startDate: subDays(startOfDay(now), 1),
          endDate: subDays(endOfDay(now), 1),
          unit: 'hour',
          value,
        };
      case 'week':
        return {
          startDate: subDays(startOfWeek(now, { locale: dateLocale }), 7),
          endDate: subDays(endOfWeek(now, { locale: dateLocale }), 1),
          unit: 'day',
          value,
        };
      case 'month':
        return {
          startDate: subMonths(startOfMonth(now), 1),
          endDate: subMonths(endOfMonth(now), 1),
          unit: 'day',
          value,
        };
      case 'year':
        return {
          startDate: subYears(startOfYear(now), 1),
          endDate: subYears(endOfYear(now), 1),
          unit: 'month',
          value,
        };
    }
  }

  switch (unit) {
    case 'day':
      return {
        startDate: subDays(startOfDay(now), num - 1),
        endDate: endOfDay(now),
        unit,
        value,
      };
    case 'hour':
      return {
        startDate: subHours(startOfHour(now), num - 1),
        endDate: endOfHour(now),
        unit,
        value,
      };
  }
}

/**
 * Calculates and returns the start and end dates of a date range along with the appropriate unit
 * @param {Date} startDate - the start date of the date range
 * @param {Date} endDate - the end date of the date range
 * @return {object} - the date range object with startDate, endDate, and unit properties
 */
export function getDateRangeValues(startDate, endDate) {
  let unit = 'year';
  if (differenceInHours(endDate, startDate) <= 48) {
    unit = 'hour';
  } else if (differenceInCalendarDays(endDate, startDate) <= 90) {
    unit = 'day';
  } else if (differenceInCalendarMonths(endDate, startDate) <= 24) {
    unit = 'month';
  }

  return { startDate: startOfDay(startDate), endDate: endOfDay(endDate), unit };
}

/**
 * Parses a datetime string and returns the corresponding date object
 * @param {string} str - the datetime string to parse
 * @return {Date} - the parsed date object
 */
export function getDateFromString(str) {
  const [ymd, hms] = str.split(' ');
  const [year, month, day] = ymd.split('-');

  if (hms) {
    const [hour, min, sec] = hms.split(':');

    return new Date(year, month - 1, day, hour, min, sec);
  }

  return new Date(year, month - 1, day);
}

/**
 * Calculates and returns an array of date objects within a date range with the given unit
 * @param {array} data - the data array to search for matches
 * @param {Date} startDate - the start date of the date range
 * @param {Date} endDate - the end date of the date range
 * @param {string} unit - the unit of time to use for the date range
 * @return {array} - the array of date objects with corresponding data values
 */
export function getDateArray(data, startDate, endDate, unit) {
  const arr = [];
  const [diff, add, normalize] = DATE_FUNCS[unit];
  const n = diff(endDate, startDate) + 1;

  /**
   * Searches the data array for a matching date and returns the corresponding y value
   * @param {Date} date - the date object to search for a match in the data array
   * @return {*} - the corresponding y value from the data array
   */
  function findData(date) {
    const d = data.find(({ x }) => {
      return normalize(getDateFromString(x)).getTime() === date.getTime();
    });

    return d?.y || 0;
  }

  for (let i = 0; i < n; i++) {
    const t = normalize(add(startDate, i));
    const y = findData(t);

    arr.push({ x: t, y });
  }

  return arr;
}

/**
 * Calculates and returns the length of a date range in the given unit
 * @param {Date} startDate - the start date of the date range
 * @param {Date} endDate - the end date of the date range
 * @param {string} unit - the unit of time to use for the date range
 * @return {number} - the length of the date range in the given unit
 */
export function getDateLength(startDate, endDate, unit) {
  const [diff] = DATE_FUNCS[unit];
  return diff(endDate, startDate) + 1;
}

/**
 * Formats a given date object with the specified format string and locale
 * @param {Date} date - the date object to format
 * @param {string} str - the format string to use
 * @param {string} locale - the locale string for the date format
 * @return {string} - the formatted date string
 */
export function dateFormat(date, str, locale = 'en-US') {
  return format(date, customFormats?.[locale]?.[str] || str, {
    locale: getDateLocale(locale),
  });
}

export const customFormats = {
  'en-US': {
    p: 'ha',
    pp: 'h:mm:ss',
  },
  'fr-FR': {
    'M/d': 'd/M',
    'MMM d': 'd MMM',
    'EEE M/d': 'EEE d/M',
  },
};

