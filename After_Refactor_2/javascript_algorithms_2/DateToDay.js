// To improve the maintainability of this codebase, we can make the following refactors:

// 1. Add comments to explain the purpose of each section of code.
// 2. Divide the code into separate functions for better readability and maintainability.
// 3. Use meaningful variable names to improve code readability.
// 4. Add error handling for invalid input.

// Here's the refactored code:

// ```
// Array holding the names of the days: Saturday - Sunday - Friday => 0 - 1 - 6
const daysNameArr = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];

// Check if the input is a valid string date
const isValidDate = (date) => {
  const [day, month, year] = date.split('/').map((x) => Number(x));
  
  // Validate the day, month, and year values
  return (
    day >= 1 && day <= 31 &&
    month >= 1 && month <= 12 &&
    !isNaN(year)
  );
};

// Calculate the day of the week using Zeller's congruence algorithm
const calculateWeekDay = (date) => {
  const [day, month, year] = date.split('/').map((x) => Number(x));

  // In case of Jan and Feb:
  // Year: we consider it as the previous year
  // e.g., 1/1/1987 here year is 1986 (-1)
  // Month: we consider the value as 13 & 14 respectively
  let adjustedYear = year;
  let adjustedMonth = month;
  if (month < 3) {
    adjustedYear--;
    adjustedMonth += 12;
  }

  // Divide the year into century and the last two digits of the century
  const yearDigits = adjustedYear % 100;
  const century = Math.floor(adjustedYear / 100);

  /*
    In mathematics, remainders of divisions are usually defined to always be positive;
    As an example, -2 mod 7 = 5.
    Many programming languages including JavaScript implement the remainder of `n % m` as `sign(n) * (abs(n) % m)`.
    This means the result has the same sign as the numerator. Here, `-2 % 7 = -1 * (2 % 7) = -2`.

    To ensure a positive numerator, the formula is adapted: `- 2 * century` is replaced with `+ 5 * century`
    which does not alter the resulting numbers mod 7 since `7 - 2 = 5`

    The following example shows the issue with modulo division:
    Without the adaption, the formula yields `weekDay = -6` for the date 2/3/2014;
    With the adaption, it yields the positive result `weekDay = 7 - 6 = 1` (Sunday), which is what we need to index the array
  */
  const weekDay = (adjustedDay + Math.floor((adjustedMonth + 1) * 2.6) + yearDigits + Math.floor(yearDigits / 4) + Math.floor(century / 4) + 5 * century) % 7;
  
  return weekDay;
};

// Get the name of the day based on the calculated weekDay
const getDayName = (weekDay) => {
  return daysNameArr[weekDay];
};

// Date to Day function
const dateToDay = (date) => {
  // Check if the input is a string
  if (typeof date !== 'string') {
    throw new TypeError('Argument is not a string.');
  }
  
  // Check if the date is valid
  if (!isValidDate(date)) {
    throw new TypeError('Date is not valid.');
  }
  
  const weekDay = calculateWeekDay(date);
  const dayName = getDayName(weekDay);
  
  return dayName;
};

export { dateToDay };
// ```

// With these refactors, the code is now more modular, readable, and follows best practices for maintainability.

