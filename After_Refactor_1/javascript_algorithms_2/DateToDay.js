// To improve the maintainability of the codebase, here are some suggested improvements:

// 1. Split the codebase into smaller and more modular functions to improve readability and reusability.
// 2. Add proper comments and documentation to explain the purpose and functionality of each function.
// 3. Use meaningful variable and function names to enhance readability.
// 4. Handle errors and exceptions gracefully by using appropriate error handling techniques.
// 5. Use consistent formatting and indentation for better code organization.
// 6. Use ES6 features such as arrow functions, destructuring, and template literals to improve code conciseness and readability.

// Here is the refactored code:

// ```javascript
// Array holding name of the day: Saturday - Sunday - Friday => 0 - 1 - 6
const daysNameArr = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

/**
 * Converts a date string to the name of the day.
 * @param {string} date - The date in string format (dd/mm/yyyy)
 * @returns {string} - The name of the day
 * @throws {TypeError} - If the input is not a string or the date is invalid
 */
const dateToDay = (date) => {
  // Check if the input is a string
  if (typeof date !== 'string') {
    throw new TypeError('Argument is not a string.')
  }

  // Extract the day, month, and year from the date string
  const [day, month, year] = date.split('/').map(Number)

  // Check if the date is valid
  if (day < 1 || day > 31 || month < 1 || month > 12) {
    throw new TypeError('Date is not valid.')
  }

  // In case of Jan and Feb, consider them as 13 and 14 respectively of the previous year
  let adjustedMonth = month
  let adjustedYear = year

  if (month < 3) {
    adjustedMonth += 12
    adjustedYear--
  }

  // Calculate the day of the week using Zeller's Congruence algorithm
  const century = Math.floor(adjustedYear / 100)
  const yearDigits = adjustedYear % 100
  
  const weekDay = (
    day +
    Math.floor((adjustedMonth + 1) * 2.6) +
    yearDigits +
    Math.floor(yearDigits / 4) +
    Math.floor(century / 4) +
    5 * century
  ) % 7

  return daysNameArr[weekDay] // Name of the weekday
}

// Example: dateToDay("18/12/2020") => "Friday"

export { dateToDay }
// ```

// By refactoring the code in this way, it becomes more maintainable, readable, and organized. Each function has a clear purpose, proper error handling is implemented, and the code follows best practices for naming conventions and readability.

