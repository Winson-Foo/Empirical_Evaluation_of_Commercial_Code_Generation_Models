// To improve the maintainability of the codebase, we can make the following changes:

// 1. Break down the long function `fillEncodeFence` into smaller functions to improve readability.
// 2. Move the type definitions out of the code to make it cleaner.
// 3. Use more descriptive variable names to improve code clarity.
// 4. Remove unnecessary comments that only restate the code.
// 5. Add error handling for unexpected inputs.
// 6. Simplify the logic in some functions for better understanding.

// Here is the refactored code:

// ```javascript
type Rail = string[];
type Fence = Rail[];

type Direction = -1 | 1;

const DIRECTIONS = { UP: -1, DOWN: 1 };

const buildFence = (rowsNum: number): Fence => Array(rowsNum).fill(null).map(() => []);

const getNextDirection = ({ railCount, currentRail, direction }: { railCount: number, currentRail: number, direction: Direction }): Direction => {
  if (currentRail === 0) {
    return DIRECTIONS.DOWN;
  } else if (currentRail === railCount - 1) {
    return DIRECTIONS.UP;
  } else {
    return direction;
  }
};

const addCharToRail = (targetRailIndex: number, letter: string) => (rail: Rail, currentRail: number): Rail =>
  currentRail === targetRailIndex ? [...rail, letter] : rail;

const fillEncodeFence = ({ fence, currentRail, direction, chars }: { fence: Fence, currentRail: number, direction: Direction, chars: string[] }): Fence => {
  if (chars.length === 0) {
    return fence;
  }

  const railCount = fence.length;
  const [letter, ...nextChars] = chars;
  const nextDirection = getNextDirection({ railCount, currentRail, direction });

  return fillEncodeFence({
    fence: fence.map(addCharToRail(currentRail, letter)),
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    chars: nextChars,
  });
};

const fillDecodeFence = ({ strLen, chars, fence, targetRail, direction, coords }: { strLen: number, chars: string[], fence: Fence, targetRail: number, direction: Direction, coords: number[] }): Fence => {
  if (chars.length === 0) {
    return fence;
  }

  const railCount = fence.length;
  const [currentRail, currentColumn] = coords;
  const shouldGoNextRail = currentColumn === strLen - 1;
  const nextDirection = shouldGoNextRail ? DIRECTIONS.DOWN : getNextDirection({ railCount, currentRail, direction });
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

const decodeFence = ({ strLen, fence, currentRail, direction, code }: { strLen: number, fence: Fence, currentRail: number, direction: Direction, code: string[] }): string => {
  if (code.length === strLen) {
    return code.join('');
  }

  const railCount = fence.length;
  const [currentChar, ...nextRail] = fence[currentRail];
  const nextDirection = getNextDirection({ railCount, currentRail, direction });

  return decodeFence({
    strLen,
    fence: fence.map((rail, idx) => (idx === currentRail ? nextRail : rail)),
    currentRail: currentRail + nextDirection,
    direction: nextDirection,
    code: [...code, currentChar],
  });
};

export const encodeRailFenceCipher = (string: string, railCount: number): string => {
  if (railCount < 2) {
    throw new Error('Rail count must be at least 2');
  }

  const fence = buildFence(railCount);
  const filledFence = fillEncodeFence({
    fence,
    currentRail: 0,
    direction: DIRECTIONS.DOWN,
    chars: string.split(''),
  });

  return filledFence.flat().join('');
};

export const decodeRailFenceCipher = (string: string, railCount: number): string => {
  if (railCount < 2) {
    throw new Error('Rail count must be at least 2');
  }

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
// ```

// With these improvements, the codebase should be more maintainable and easier to understand.

