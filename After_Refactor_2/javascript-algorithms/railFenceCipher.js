// To improve the maintainability of the codebase, the following changes can be made:

// 1. Provide meaningful names for variables and functions to improve code readability.

// 2. Use comments to explain the purpose and functionality of each function.

// 3. Separate the code into smaller, reusable functions to improve modularity and readability.

// 4. Remove unnecessary comments and simplify complex logic.

// Here is the refactored code:

// ```
/**
 * @typedef {string[]} Rail
 * @typedef {Rail[]} Fence
 * @typedef {number} Direction
 */

/**
 * Enum defining the possible directions.
 */
const DIRECTIONS = {
  UP: -1,
  DOWN: 1,
};

/**
 * Builds a fence with a specific number of rows.
 *
 * @param {number} rowsNum - Number of rows in the fence
 * @returns {Fence} - The built fence
 */
const buildFence = (rowsNum) => Array(rowsNum).fill(null).map(() => []);

/**
 * Get the next direction to move while traversing the fence.
 *
 * @param {object} params - Parameters
 * @param {number} params.railCount - Number of rows in the fence
 * @param {number} params.currentRail - Current row that we're visiting
 * @param {Direction} params.direction - Current direction
 * @returns {Direction} - The next direction to take
 */
const getNextDirection = ({ railCount, currentRail, direction }) => {
  if (currentRail === 0) {
    return DIRECTIONS.DOWN;
  } else if (currentRail === railCount - 1) {
    return DIRECTIONS.UP;
  } else {
    return direction;
  }
};

/**
 * Adds a character to a rail if it matches the target index.
 *
 * @param {number} targetRailIndex - Target rail index
 * @param {string} letter - The character to add
 * @returns {Function} - The onEachRail function
 */
const addCharToRail = (targetRailIndex, letter) => {
  /**
   * Given a rail, adds a char to it if it matches the targetIndex.
   *
   * @param {Rail} rail - The current rail
   * @param {number} currentRail - The index of the current rail
   * @returns {Rail} - The modified rail
   */
  function onEachRail(rail, currentRail) {
    return currentRail === targetRailIndex ? [...rail, letter] : rail;
  }
  return onEachRail;
};

/**
 * Fills the fence with the characters.
 *
 * @param {object} params - Parameters
 * @param {Fence} params.fence - The fence
 * @param {number} params.currentRail - Current rail index
 * @param {Direction} params.direction - Current direction
 * @param {string[]} params.chars - The characters to place on the fence
 * @returns {Fence} - The filled fence
 */
const fillEncodeFence = ({
  fence,
  currentRail,
  direction,
  chars,
}) => {
  if (chars.length === 0) {
    return fence;
  }

  const railCount = fence.length;
  const [letter, ...nextChars] = chars;
  const nextDirection = getNextDirection({
    railCount,
    currentRail,
    direction,
  });

  return fillEncodeFence({
    fence: fence.map(addCharToRail(currentRail, letter)),
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    chars: nextChars,
  });
};

/**
 * Fills the fence with the characters during decoding.
 *
 * @param {object} params - Parameters
 * @param {number} params.strLen - Length of the string
 * @param {string[]} params.chars - The characters to place on the fence
 * @param {Fence} params.fence - The fence
 * @param {number} params.targetRail - Target rail index
 * @param {Direction} params.direction - Current direction
 * @param {number[]} params.coords - Coordinates
 * @returns {Fence} - The filled fence
 */
const fillDecodeFence = (params) => {
  const {
    strLen, chars, fence, targetRail, direction, coords,
  } = params;

  const railCount = fence.length;

  if (chars.length === 0) {
    return fence;
  }

  const [currentRail, currentColumn] = coords;
  const shouldGoNextRail = currentColumn === strLen - 1;
  const nextDirection = shouldGoNextRail
    ? DIRECTIONS.DOWN
    : getNextDirection({ railCount, currentRail, direction });
  const nextRail = shouldGoNextRail ? targetRail + 1 : targetRail;
  const nextCoords = [
    shouldGoNextRail ? 0 : currentRail + nextDirection,
    shouldGoNextRail ? 0 : currentColumn + 1,
  ];

  const shouldAddChar = currentRail === targetRail;
  const [currentChar, ...remainderChars] = chars;
  const nextString = shouldAddChar ? remainderChars : chars;
  const nextFence = shouldAddChar ? fence.map(addCharToRail(currentRail, currentChar)) : fence;

  return fillDecodeFence({
    strLen,
    chars: nextString,
    fence: nextFence,
    targetRail: nextRail,
    direction: nextDirection,
    coords: nextCoords,
  });
};

/**
 * Decodes the message using Rail Fence Cipher.
 *
 * @param {object} params - Parameters
 * @param {number} params.strLen - Length of the string
 * @param {Fence} params.fence - The fence
 * @param {number} params.currentRail - Current rail index
 * @param {Direction} params.direction - Current direction
 * @param {number[]} params.code - The decoded characters
 * @returns {string} - The decoded string
 */
const decodeFence = (params) => {
  const {
    strLen,
    fence,
    currentRail,
    direction,
    code,
  } = params;

  if (code.length === strLen) {
    return code.join('');
  }

  const railCount = fence.length;

  const [currentChar, ...nextRail] = fence[currentRail];
  const nextDirection = getNextDirection({ railCount, currentRail, direction });

  return decodeFence({
    railCount,
    strLen,
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    code: [...code, currentChar],
    fence: fence.map((rail, idx) => (idx === currentRail ? nextRail : rail)),
  });
};

/**
 * Encodes the message using Rail Fence Cipher.
 *
 * @param {string} string - The string to be encoded
 * @param {number} railCount - The number of rails in a fence
 * @returns {string} - The encoded string
 */
export const encodeRailFenceCipher = (string, railCount) => {
  const fence = buildFence(railCount);

  const filledFence = fillEncodeFence({
    fence,
    currentRail: 0,
    direction: DIRECTIONS.DOWN,
    chars: string.split(''),
  });

  return filledFence.flat().join('');
};

/**
 * Decodes the message using Rail Fence Cipher.
 *
 * @param {string} string - The encoded string
 * @param {number} railCount - The number of rows in a fence
 * @returns {string} - The decoded string.
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
    coords: [0, 0],
  });

  return decodeFence({
    strLen,
    fence: filledFence,
    currentRail: 0,
    direction: DIRECTIONS.DOWN,
    code: [],
  });
}; 

