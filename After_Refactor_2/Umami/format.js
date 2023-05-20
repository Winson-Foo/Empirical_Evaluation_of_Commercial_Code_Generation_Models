/**
 * Parses a time value (in seconds) into its individual components (days, hours, minutes, seconds, and milliseconds).
 */
export function parseTime(val) {
  const days = Math.floor(val / 86400);
  const hours = Math.floor((val % 86400) / 3600);
  const minutes = Math.floor((val % 3600) / 60);
  const seconds = Math.floor(val % 60);
  const ms = Math.round((val - Math.floor(val)) * 1000);

  return { days, hours, minutes, seconds, ms };
}

/**
 * Formats a time value (in seconds) into a string in the format "H:MM:SS".
 */
export function formatTime(val) {
  const { hours, minutes, seconds } = parseTime(val);
  const h = hours > 0 ? `${hours}:` : '';
  const m = hours > 0 ? minutes.toString().padStart(2, '0') : minutes;
  const s = seconds.toString().padStart(2, '0');

  return `${h}${m}:${s}`;
}

/**
 * Formats a time value (in seconds) into a short string representation with configurable formats ("d", "h", "m", "s", "ms").
 */
export function formatTimeShort(val, formats = ['m', 's'], space = '') {
  const { days, hours, minutes, seconds, ms } = parseTime(val);

  let t = '';

  if (days > 0 && formats.includes('d')) t += `${days}d${space}`;
  if (hours > 0 && formats.includes('h')) t += `${hours}h${space}`;
  if (minutes > 0 && formats.includes('m')) t += `${minutes}m${space}`;
  if (seconds > 0 && formats.includes('s')) t += `${seconds}s${space}`;
  if (ms > 0 && formats.includes('ms')) t += `${ms}ms`;

  if (!t) {
    return `0${formats[formats.length - 1]}`;
  }

  return t;
}

/**
 * Formats a number with no decimal places.
 */
export function formatNumberFixed(n) {
  return Number(n).toFixed(0);
}

/**
 * Formats a number with a variable number of decimal places and appends a "k" or "m" suffix if the value is above a certain threshold.
 */
export function formatLongNumber(value) {
  const n = Number(value);

  const entry = THRESHOLDS.find((entry) => n >= entry.threshold);

  if (entry) {
    return `${moveDecimals(n, entry.precision)}${entry.divisor === 1000 ? 'k' : 'm'}`;
  }

  return formatNumberFixed(n);
}

/**
 * Generates a hexadecimal color code from a string value.
 */
export function stringToColor(str) {
  if (!str) {
    return '#ffffff';
  }

  let hash = 0;

  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  let color = '#';

  for (let i = 0; i < 3; i++) {
    let value = (hash >> (i * 8)) & 0xff;
    color += ('00' + value.toString(16)).slice(-2);
  }

  return color;
}