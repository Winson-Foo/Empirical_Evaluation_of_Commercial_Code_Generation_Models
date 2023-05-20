const SECONDS_PER_DAY = 86400;
const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;

export function parseTime(timeInSeconds) {
  const days = calculateDays(timeInSeconds);
  const hours = calculateHours(timeInSeconds);
  const minutes = calculateMinutes(timeInSeconds);
  const seconds = calculateSeconds(timeInSeconds);
  const ms = calculateMilliseconds(timeInSeconds);

  return {
    days,
    hours,
    minutes,
    seconds,
    ms,
  };
}

function calculateDays(timeInSeconds) {
  return Math.floor(timeInSeconds / SECONDS_PER_DAY);
}

function calculateHours(timeInSeconds) {
  return Math.floor(timeInSeconds / SECONDS_PER_HOUR) - calculateDays(timeInSeconds) * 24;
}

function calculateMinutes(timeInSeconds) {
  return Math.floor(timeInSeconds / SECONDS_PER_MINUTE) - calculateDays(timeInSeconds) * 1440 - calculateHours(timeInSeconds) * 60;
}

function calculateSeconds(timeInSeconds) {
  return Math.floor(timeInSeconds) - calculateDays(timeInSeconds) * SECONDS_PER_DAY - calculateHours(timeInSeconds) * SECONDS_PER_HOUR - calculateMinutes(timeInSeconds) * SECONDS_PER_MINUTE;
}

function calculateMilliseconds(timeInSeconds) {
  return Math.round((timeInSeconds - Math.floor(timeInSeconds)) * 1000);
}

export function formatTime(timeInSeconds) {
  const { hours, minutes, seconds } = parseTime(timeInSeconds);
  const h = hours > 0 ? `${hours}:` : '';
  const m = hours > 0 ? minutes.toString().padStart(2, '0') : minutes;
  const s = seconds.toString().padStart(2, '0');

  return `${h}${m}:${s}`;
}

export function formatValue(value, options = {}) {
  const { suffix } = options;
  const { days, hours, minutes, seconds, ms } = parseTime(value);
  let formattedValue = '';

  if (days > 0) {
    formattedValue = `${days}d `;
  }
  if (hours > 0) {
    formattedValue += `${hours}h `;
  }
  if (minutes > 0) {
    formattedValue += `${minutes}m `;
  }
  if (seconds > 0) {
    formattedValue += `${seconds}s `;
  }
  if (ms > 0) {
    formattedValue += `${ms}ms `;
  }
  if (formattedValue === '') {
    formattedValue = `${value.toFixed(0)}${suffix}`;
  } else {
    formattedValue = formattedValue.trim();
    if (suffix) {
      formattedValue += ` ${suffix}`;
    }
  }

  return formattedValue;
}

export function formatNumber(number) {
  return Number(number).toFixed(0);
}

export function formatLongNumber(number) {
  const n = Number(number);

  if (n >= 1000000) {
    return `${(n / 1000000).toFixed(1)}m`;
  }
  if (n >= 100000) {
    return `${(n / 1000).toFixed(0)}k`;
  }
  if (n >= 10000) {
    return `${(n / 1000).toFixed(1)}k`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(2)}k`;
  }

  return formatNumber(n);
}

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

