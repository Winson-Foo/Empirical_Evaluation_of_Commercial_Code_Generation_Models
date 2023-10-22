// To improve the maintainability of this codebase, we can make the following changes:

// 1. Replace the use of constant values with descriptive variables and make the code more readable.

// 2. Break down the long functions into smaller, more focused functions.

// 3. Add comments to explain the purpose and functionality of each function.

// 4. Use destructuring syntax to improve code readability and reduce the number of parameters passed.

// Here is the refactored code:

// ```javascript
/**
 * @typedef {string[]} Rail
 * @typedef {Rail[]} Fence
 * @typedef {number} Direction
 */

/**
 * @constant DIRECTIONS
 * @type {object}
 * @property {Direction} UP
 * @property {Direction} DOWN
 */
const DIRECTIONS = {
  UP: -1,
  DOWN: 1
};

/**
 * Builds a fence with a specific number of rows.
 *
 * @param {number} rowsNum
 * @returns {Fence}
 */
const buildFence = (rowsNum) => Array(rowsNum).fill(null).map(() => []);

/**
 * Get the next direction to move (based on the current one) while traversing the fence.
 *
 * @param {object} params
 * @param {number} params.railCount - Number of rows in the fence
 * @param {number} params.currentRail - Current row that we're visiting
 * @param {Direction} params.direction - Current direction
 * @returns {Direction} - The next direction to take
 */
const getNextDirection = ({ railCount, currentRail, direction }) => {
  switch (currentRail) {
    case 0:
      // Go down if we're at the top of the fence.
      return DIRECTIONS.DOWN;
    case railCount - 1:
      // Go up if we're at the bottom of the fence.
      return DIRECTIONS.UP;
    default:
      // Continue with the same direction if we're in the middle of the fence.
      return direction;
  }
};

/**
 * Given a rail, adds a char to it if it matches the targetIndex.
 *
 * @param {number} targetRailIndex
 * @param {string} letter
 * @param {Rail} rail
 * @param {number} currentRail
 * @returns {Rail}
 */
const addCharToRail = (targetRailIndex, letter, rail, currentRail) => {
  return currentRail === targetRailIndex ? [...rail, letter] : rail;
};

/**
 * Hangs the characters on the fence.
 *
 * @param {object} params
 * @param {Fence} params.fence
 * @param {number} params.currentRail
 * @param {Direction} params.direction
 * @param {string[]} params.chars
 * @returns {Fence}
 */
const fillEncodeFence = ({
  fence,
  currentRail,
  direction,
  chars
}) => {
  if (chars.length === 0) {
    // All chars have been placed on the fence.
    return fence;
  }

  const railCount = fence.length;
  const [letter, ...nextChars] = chars;
  const nextDirection = getNextDirection({
    railCount,
    currentRail,
    direction
  });

  const nextFence = fence.map((rail, idx) =>
    addCharToRail(currentRail, letter, rail, idx)
  );

  return fillEncodeFence({
    fence: nextFence,
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    chars: nextChars
  });
};

/**
 * Hangs the characters on the fence while decoding.
 *
 * @param {object} params
 * @param {number} params.strLen
 * @param {string[]} params.chars
 * @param {Fence} params.fence
 * @param {number} params.targetRail
 * @param {Direction} params.direction
 * @param {number[]} params.coords
 * @returns {Fence}
 */
const fillDecodeFence = (params) => {
  const {
    strLen,
    chars,
    fence,
    targetRail,
    direction,
    coords
  } = params;

  if (chars.length === 0) {
    return fence;
  }

  const railCount = fence.length;
  const [currentRail, currentColumn] = coords;

  const shouldGoNextRail = currentColumn === strLen - 1;
  const nextDirection = shouldGoNextRail
    ? DIRECTIONS.DOWN
    : getNextDirection({ railCount, currentRail, direction });

  const nextRail = shouldGoNextRail ? targetRail + 1 : targetRail;
  const nextCoords = [
    shouldGoNextRail ? 0 : currentRail + nextDirection,
    shouldGoNextRail ? 0 : currentColumn + 1
  ];

  const shouldAddChar = currentRail === targetRail;
  const [currentChar, ...remainderChars] = chars;
  const nextString = shouldAddChar ? remainderChars : chars;

  const nextFence = shouldAddChar
    ? fence.map((rail, idx) =>
        addCharToRail(currentRail, currentChar, rail, idx)
      )
    : fence;

  return fillDecodeFence({
    strLen,
    chars: nextString,
    fence: nextFence,
    targetRail: nextRail,
    direction: nextDirection,
    coords: nextCoords
  });
};

/**
 * Decodes the fence to retrieve the original message.
 *
 * @param {object} params
 * @param {number} params.strLen
 * @param {Fence} params.fence
 * @param {number} params.currentRail
 * @param {Direction} params.direction
 * @param {number[]} params.code
 * @returns {string}
 */
const decodeFence = (params) => {
  const {
    strLen,
    fence,
    currentRail,
    direction,
    code
  } = params;

  if (code.length === strLen) {
    return code.join('');
  }

  const railCount = fence.length;
  const [currentChar, ...nextRail] = fence[currentRail];
  const nextDirection = getNextDirection({
    railCount,
    currentRail,
    direction
  });

  const nextFence = fence.map((rail, idx) =>
    idx === currentRail ? nextRail : rail
  );

  return decodeFence({
    strLen,
    fence: nextFence,
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    code: [...code, currentChar]
  });
};

/**
 * Encodes the message using Rail Fence Cipher.
 *
 * @param {string} string - The string to be encoded
 * @param {number} railCount - The number of rails in a fence
 * @returns {string} - Encoded string
 */
export const encodeRailFenceCipher = (string, railCount) => {
  const fence = buildFence(railCount);

  const filledFence = fillEncodeFence({
    fence,
    currentRail: 0,
    direction: DIRECTIONS.DOWN,
    chars: string.split('')
  });

  return filledFence.flat().join('');
};

/**
 * Decodes the message using Rail Fence Cipher.
 *
 * @param {string} string - Encoded string
 * @param {number} railCount - The number of rows in a fence
 * @returns {string} - Decoded string.
 */
export const decodeRailFenceCipher = (string, railCount) => {
  const strLen = string.length;
  const emptyFence = buildFence(railCount);
  const filledFence = fillDecodeFence({
    strLen,
    chars: string.split(''),
    fence: emptyFence,
    targetRail: 0,
    direction: DIRECTIONS.DOWN,
    coords: [0, 0]
  });

  return decodeFence({
    strLen,
    fence: filledFence,
    currentRail: 0,
    direction: DIRECTIONS.DOWN,
    code: []
  });
}; 

