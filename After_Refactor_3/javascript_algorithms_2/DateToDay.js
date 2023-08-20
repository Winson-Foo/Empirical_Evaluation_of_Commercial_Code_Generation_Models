// To improve the maintainability of the codebase, we can make the following changes:

// 1. Split the logic into smaller functions: Breaking down the code into smaller functions can make it easier to understand and maintain. We can separate the date validation logic and the algorithm calculation into separate functions.

// ```javascript
const isValidDate = (day, month) => {
  return day >= 1 && day <= 31 && month >= 1 && month <= 12;
};

const calculateWeekDay = (day, month, year) => {
  if (month < 3) {
    year--;
    month += 12;
  }

  const yearDigits = year % 100;
  const century = Math.floor(year / 100);

  const weekDay =
    (day +
      Math.floor((month + 1) * 2.6) +
      yearDigits +
      Math.floor(yearDigits / 4) +
      Math.floor(century / 4) +
      5 * century) %
    7;
  return weekDay;
};

const getDayName = (weekDay) => {
  const daysNameArr = [
    'Saturday',
    'Sunday',
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
  ];
  return daysNameArr[weekDay];
};

const formatDateToDay = (dateString) => {
  if (typeof dateString !== 'string') {
    throw new TypeError('Argument is not a string.');
  }

  const [day, month, year] = dateString.split('/').map(Number);

  if (!isValidDate(day, month)) {
    throw new TypeError('Date is not valid.');
  }

  const weekDay = calculateWeekDay(day, month, year);
  return getDayName(weekDay);
};

export { formatDateToDay };
// ```

// 2. Add error handling: Instead of returning a TypeError, we can throw an error using the `throw` keyword. This allows for better error handling and makes it clear that an error has occurred.

// 3. Use more descriptive function and variable names: By using more descriptive names for functions and variables, the code becomes easier to understand. This helps in maintaining and debugging the code in the future.

// By refactoring the code in this way, we have improved the maintainability by breaking down the logic into smaller functions, adding error handling, and using more descriptive names for functions and variables.

