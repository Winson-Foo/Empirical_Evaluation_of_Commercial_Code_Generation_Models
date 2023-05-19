export function getTimeBreakdown(timeInSeconds) {
  const secondsInDay = 86400;
  const secondsInHour = 3600;
  const secondsInMinute = 60;

  const days = ~~(timeInSeconds / secondsInDay);
  const remainingSeconds = timeInSeconds - (days * secondsInDay);
  const hours = ~~(remainingSeconds / secondsInHour);
  const remainingSecondsAfterHours = remainingSeconds - (hours * secondsInHour);
  const minutes = ~~(remainingSecondsAfterHours / secondsInMinute);
  const seconds = remainingSecondsAfterHours - (minutes * secondsInMinute);
  const milliseconds = (timeInSeconds - ~~timeInSeconds) * 1000;

  return {
    days,
    hours,
    minutes,
    seconds,
    milliseconds,
  };
}

export function formatTime(timeInSeconds) {
  const { hours, minutes, seconds } = getTimeBreakdown(timeInSeconds);
  const hourString = hours > 0 ? `${hours}:` : '';
  const minuteString = hours > 0 ? minutes.toString().padStart(2, '0') : minutes;
  const secondString = seconds.toString().padStart(2, '0');

  return `${hourString}${minuteString}:${secondString}`;
}

export function formatShortTime(timeInSeconds, formats = ['m', 's'], space = '') {
  const { days, hours, minutes, seconds, milliseconds } = getTimeBreakdown(timeInSeconds);
  let timeString = '';

  if (days > 0 && formats.indexOf('d') !== -1) timeString += `${days}d${space}`;
  if (hours > 0 && formats.indexOf('h') !== -1) timeString += `${hours}h${space}`;
  if (minutes > 0 && formats.indexOf('m') !== -1) timeString += `${minutes}m${space}`;
  if (seconds > 0 && formats.indexOf('s') !== -1) timeString += `${seconds}s${space}`;
  if (milliseconds > 0 && formats.indexOf('ms') !== -1) timeString += `${milliseconds}ms`;

  if (!timeString) {
    return `0${formats[formats.length - 1]}`;
  }

  return timeString;
}

export function formatNumber(number) {
  return Number(number).toFixed(0);
}

export function formatLongNumber(value) {
  const number = Number(value);
  const million = 1000000;
  const thousand = 1000;

  if (number >= million) {
    return `${(number / million).toFixed(1)}m`;
  }
  if (number >= thousand * 10) {
    return `${(number / thousand).toFixed(0)}k`;
  }
  if (number >= thousand) {
    return `${(number / thousand).toFixed(1)}k`;
  }
  if (number >= 1) {
    return `${(number / 1000).toFixed(2)}k`;
  }

  return formatNumber(number);
}

export function stringToColor(str) {
  const defaultColor = '#ffffff';
  if (!str) {
    return defaultColor;
  }

  const hash = Array.from(str).reduce((hashAccumulator, character) =>
    character.charCodeAt(0) + ((hashAccumulator << 5) - hashAccumulator), 0);

  let color = '#';

  for (let i = 0; i < 3; i++) {
    const colorValue = (hash >> (i * 8)) & 0xff;
    color += ('00' + colorValue.toString(16)).slice(-2);
  }

  return color || defaultColor;
} 