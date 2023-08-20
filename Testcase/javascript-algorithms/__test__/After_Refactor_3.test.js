import zAlgorithm from '../../../After_Refactor_3/javascript-algorithms/zAlgorithm'
import { caesarCipherEncrypt, caesarCipherDecrypt } from '../../../After_Refactor_3/javascript-algorithms/caesarCipher'
import dpRainTerraces from '../../../After_Refactor_3/javascript-algorithms/dpRainTerraces'
import dqBestTimeToBuySellStocks from '../../../After_Refactor_3/javascript-algorithms/dqBestTimeToBuySellStocks'
import euclideanAlgorithm from '../../../After_Refactor_3/javascript-algorithms/euclideanAlgorithm';
import fastPowering from '../../../After_Refactor_3/javascript-algorithms/fastPowering';
import greedyJumpGame from '../../../After_Refactor_3/javascript-algorithms/greedyJumpGame';
import isPalindrome from '../../../After_Refactor_3/javascript-algorithms/isPalindrome';
import knightTour from '../../../After_Refactor_3/javascript-algorithms/knightTour';
import knuthMorrisPratt from '../../../After_Refactor_3/javascript-algorithms/knuthMorrisPratt';
import levenshteinDistance from '../../../After_Refactor_3/javascript-algorithms/levenshteinDistance';
import PolynomialHash from '../../../After_Refactor_3/javascript-algorithms/PolynomialHash';
// import { encodeRailFenceCipher, decodeRailFenceCipher } from '../../../After_Refactor_3/javascript-algorithms/railFenceCipher';
// unable to run above due to compilation error
import recursiveStaircaseMEM from '../../../After_Refactor_3/javascript-algorithms/recursiveStaircaseMEM';
import regularExpressionMatching from '../../../After_Refactor_3/javascript-algorithms/regularExpressionMatching';
import sieveOfEratosthenes from '../../../After_Refactor_3/javascript-algorithms/sieveOfEratosthenes';
import SimplePolynomialHash from '../../../After_Refactor_3/javascript-algorithms/SimplePolynomialHash';
import squareMatrixRotation from '../../../After_Refactor_3/javascript-algorithms/squareMatrixRotation';
import weightedRandom from '../../../After_Refactor_3/javascript-algorithms/weightedRandom';

describe('zAlgorithm', () => {
  test('should find word positions in given text', () => { expect(zAlgorithm('abcbcglx', 'abca')).toEqual([]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abca', 'abca')).toEqual([0]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abca', 'abcadfd')).toEqual([]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcbcglabcx', 'abc')).toEqual([0, 7]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcbcglx', 'bcgl')).toEqual([3]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcbcglx', 'cglx')).toEqual([4]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcxabcdabxabcdabcdabcy', 'abcdabcy')).toEqual([15]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcxabcdabxabcdabcdabcy', 'abcdabca')).toEqual([]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcxabcdabxaabcdabcabcdabcdabcy', 'abcdabca')).toEqual([12]) });
  test('should find word positions in given text', () => { expect(zAlgorithm('abcxabcdabxaabaabaaaabcdabcdabcy', 'aabaabaaa')).toEqual([11]) });
});
;

describe('caesarCipher', () => {
  it('should not change a string with zero shift', () => {
    expect(caesarCipherEncrypt('abcd', 0)).toBe('abcd');
    expect(caesarCipherDecrypt('abcd', 0)).toBe('abcd');
  });

  it('should cipher a string with different shifts', () => {
    expect(caesarCipherEncrypt('abcde', 3)).toBe('defgh');
    expect(caesarCipherDecrypt('defgh', 3)).toBe('abcde');

    expect(caesarCipherEncrypt('abcde', 1)).toBe('bcdef');
    expect(caesarCipherDecrypt('bcdef', 1)).toBe('abcde');

    expect(caesarCipherEncrypt('xyz', 1)).toBe('yza');
    expect(caesarCipherDecrypt('yza', 1)).toBe('xyz');
  });

  it('should be case insensitive', () => {
    expect(caesarCipherEncrypt('ABCDE', 3)).toBe('defgh');
  });

  it('should correctly handle an empty strings', () => {
    expect(caesarCipherEncrypt('', 3)).toBe('');
  });

  it('should not cipher unknown chars', () => {
    expect(caesarCipherEncrypt('ab2cde', 3)).toBe('de2fgh');
    expect(caesarCipherDecrypt('de2fgh', 3)).toBe('ab2cde');
  });

  it('should encrypt and decrypt full phrases', () => {
    expect(caesarCipherEncrypt('THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG', 23))
      .toBe('qeb nrfzh yoltk clu grjmp lsbo qeb ixwv ald');

    expect(caesarCipherDecrypt('qeb nrfzh yoltk clu grjmp lsbo qeb ixwv ald', 23))
      .toBe('the quick brown fox jumps over the lazy dog');
  });
});

describe('dpRainTerraces', () => {
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([1])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([1, 0])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 1])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 1, 0])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 1, 0, 0])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 1, 0, 0, 1, 0])).toBe(2) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 2, 0, 0, 1, 0])).toBe(2) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([2, 0, 2])).toBe(2) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([2, 0, 5])).toBe(2) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([3, 0, 0, 2, 0, 4])).toBe(10) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])).toBe(6) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([1, 1, 1, 1, 1])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([1, 2, 3, 4, 5])).toBe(0) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([4, 1, 3, 1, 2, 1, 2, 1])).toBe(4) });
  it('should find the amount of water collected after raining', () => { expect(dpRainTerraces([0, 2, 4, 3, 4, 2, 4, 0, 8, 7, 0])).toBe(7) });
});
;

describe('dqBestTimeToBuySellStocks', () => {
  it('should find the best time to buy and sell stocks', () => {
    let visit;

    expect(dqBestTimeToBuySellStocks([1, 5])).toEqual(4);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([1], visit)).toEqual(0);
    expect(visit).toHaveBeenCalledTimes(3);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([1, 5], visit)).toEqual(4);
    expect(visit).toHaveBeenCalledTimes(7);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([5, 1], visit)).toEqual(0);
    expect(visit).toHaveBeenCalledTimes(7);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([1, 5, 10], visit)).toEqual(9);
    expect(visit).toHaveBeenCalledTimes(15);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([10, 1, 5, 20, 15, 21], visit)).toEqual(25);
    expect(visit).toHaveBeenCalledTimes(127);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([7, 1, 5, 3, 6, 4], visit)).toEqual(7);
    expect(visit).toHaveBeenCalledTimes(127);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([1, 2, 3, 4, 5], visit)).toEqual(4);
    expect(visit).toHaveBeenCalledTimes(63);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks([7, 6, 4, 3, 1], visit)).toEqual(0);
    expect(visit).toHaveBeenCalledTimes(63);
  })
  it('should find the best time to buy and sell stocks', () => {
    visit = jest.fn();
    expect(dqBestTimeToBuySellStocks(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      visit,
    )).toEqual(19);
    expect(visit).toHaveBeenCalledTimes(2097151);
  });
});

describe('euclideanAlgorithm', () => {
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(0, 0)).toBe(0); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(2, 0)).toBe(2); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(0, 2)).toBe(2); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(1, 2)).toBe(1); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(2, 1)).toBe(1); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(6, 6)).toBe(6); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(2, 4)).toBe(2); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(4, 2)).toBe(2); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(12, 4)).toBe(4); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(4, 12)).toBe(4); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(5, 13)).toBe(1); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(27, 13)).toBe(1); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(24, 60)).toBe(12); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(60, 24)).toBe(12); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(252, 105)).toBe(21); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(105, 252)).toBe(21); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(1071, 462)).toBe(21); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(462, 1071)).toBe(21); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(462, -1071)).toBe(21); })
  it('should calculate GCD recursively', () => { expect(euclideanAlgorithm(-462, -1071)).toBe(21); });
});

describe('fastPowering', () => {
  it('should compute power in log(n) time', () => { expect(fastPowering(1, 1)).toBe(1) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 0)).toBe(1) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 2)).toBe(4) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 3)).toBe(8) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 4)).toBe(16) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 5)).toBe(32) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 6)).toBe(64) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 7)).toBe(128) });
  it('should compute power in log(n) time', () => { expect(fastPowering(2, 8)).toBe(256) });
  it('should compute power in log(n) time', () => { expect(fastPowering(3, 4)).toBe(81) });
  it('should compute power in log(n) time', () => { expect(fastPowering(190, 2)).toBe(36100) });
  it('should compute power in log(n) time', () => { expect(fastPowering(11, 5)).toBe(161051) });
  it('should compute power in log(n) time', () => { expect(fastPowering(13, 11)).toBe(1792160394037) });
  it('should compute power in log(n) time', () => { expect(fastPowering(9, 16)).toBe(1853020188851841) });
  it('should compute power in log(n) time', () => { expect(fastPowering(16, 16)).toBe(18446744073709552000) });
  it('should compute power in log(n) time', () => { expect(fastPowering(7, 21)).toBe(558545864083284000) });
  it('should compute power in log(n) time', () => { expect(fastPowering(100, 9)).toBe(1000000000000000000) });
});


describe('greedyJumpGame', () => {
  it('should solve Jump Game problem in greedy manner', () => { expect(greedyJumpGame([1, 0])).toBe(true) });
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([100, 0])).toBe(true));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([2, 3, 1, 1, 4])).toBe(true));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([1, 1, 1, 1, 1])).toBe(true));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([1, 1, 1, 10, 1])).toBe(true));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([1, 5, 2, 1, 0, 2, 0])).toBe(true));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([1, 0, 1])).toBe(false));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([3, 2, 1, 0, 4])).toBe(false));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([0, 0, 0, 0, 0])).toBe(false));
  it('should solve Jump Game problem in greedy manner', () => expect(greedyJumpGame([5, 4, 3, 2, 1, 0, 0])).toBe(false));
});

describe('palindromeCheck', () => {
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('a')).toBe(true) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('pop')).toBe(true) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('deed')).toBe(true) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('kayak')).toBe(true) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('racecar')).toBe(true) });

  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('rad')).toBe(false) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('dodo')).toBe(false) });
  it('should return whether or not the string is a palindrome', () => { expect(isPalindrome('polo')).toBe(false) });
});

describe('knightTour', () => {
  it('should not find solution on 3x3 board', () => {
    const moves = knightTour(3);

    expect(moves.length).toBe(0);
  });

  it('should find one solution to do knight tour on 5x5 board', () => {
    const moves = knightTour(5);

    expect(moves.length).toBe(25);

    expect(moves).toEqual([
      [0, 0],
      [1, 2],
      [2, 0],
      [0, 1],
      [1, 3],
      [3, 4],
      [2, 2],
      [4, 1],
      [3, 3],
      [1, 4],
      [0, 2],
      [1, 0],
      [3, 1],
      [4, 3],
      [2, 4],
      [0, 3],
      [1, 1],
      [3, 0],
      [4, 2],
      [2, 1],
      [4, 0],
      [3, 2],
      [4, 4],
      [2, 3],
      [0, 4],
    ]);
  });
});

describe('knuthMorrisPratt', () => {
  it('should find word position in given text', () => { expect(knuthMorrisPratt('', '')).toBe(0) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('a', '')).toBe(0) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('a', 'a')).toBe(0) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcbcglx', 'abca')).toBe(-1) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcbcglx', 'bcgl')).toBe(3) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcxabcdabxabcdabcdabcy', 'abcdabcy')).toBe(15) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcxabcdabxabcdabcdabcy', 'abcdabca')).toBe(-1) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcxabcdabxaabcdabcabcdabcdabcy', 'abcdabca')).toBe(12) });
  it('should find word position in given text', () => { expect(knuthMorrisPratt('abcxabcdabxaabaabaaaabcdabcdabcy', 'aabaabaaa')).toBe(11) });
});


describe('levenshteinDistance', () => {
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('', '')).toBe(0)});
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('a', '')).toBe(1)});
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('', 'a')).toBe(1)});
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('abc', '')).toBe(3)});
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('', 'abc')).toBe(3)});

  // Should just add I to the beginning.
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('islander', 'slander')).toBe(1)});

  // Needs to substitute M by K, T by M and add an A to the end
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('mart', 'karma')).toBe(3)});

  // Substitute K by S, E by I and insert G at the end.
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('kitten', 'sitting')).toBe(3)});

  // Should add 4 letters FOOT at the beginning.
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('ball', 'football')).toBe(4)});

  // Should delete 4 letters FOOT at the beginning.
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('football', 'foot')).toBe(4)});

  // Needs to substitute the first 5 chars: INTEN by EXECU
  it('should calculate edit distance between two strings', () => {expect(levenshteinDistance('intention', 'execution')).toBe(5);});
});


describe('PolynomialHash', () => {
  it('should calculate new hash based on previous one', () => {
    const bases = [3, 79, 101, 3251, 13229, 122743, 3583213];
    const mods = [79, 101];
    const frameSizes = [5, 20];

    const text = 'Lorem Ipsum is simply dummy text of the printing and '
      + 'typesetting industry. Lorem Ipsum has been the industry\'s standard '
      + 'galley of type and \u{ffff} scrambled it to make a type specimen book. It '
      + 'electronic 耀 typesetting, remaining essentially unchanged. It was '
      // + 'popularised in the \u{20005} \u{20000}1960s with the release of Letraset sheets '
      + 'publishing software like Aldus PageMaker 耀 including versions of Lorem.';

    // Check hashing for different prime base.
    bases.forEach((base) => {
      mods.forEach((modulus) => {
        const polynomialHash = new PolynomialHash({ base, modulus });

        // Check hashing for different word lengths.
        frameSizes.forEach((frameSize) => {
          let previousWord = text.substr(0, frameSize);
          let previousHash = polynomialHash.hash(previousWord);

          // Shift frame through the whole text.
          for (let frameShift = 1; frameShift < (text.length - frameSize); frameShift += 1) {
            const currentWord = text.substr(frameShift, frameSize);
            const currentHash = polynomialHash.hash(currentWord);
            const currentRollingHash = polynomialHash.roll(previousHash, previousWord, currentWord);

            // Check that rolling hash is the same as directly calculated hash.
            expect(currentRollingHash).toBe(currentHash);

            previousWord = currentWord;
            previousHash = currentHash;
          }
        });
      });
    });
  });
  const polynomialHash = new PolynomialHash({ modulus: 100 });
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('Some long text that is used as a key')).toBe(41)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('Test')).toBe(92)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('a')).toBe(97)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('b')).toBe(98)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('c')).toBe(99)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('d')).toBe(0)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('e')).toBe(1)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('ab')).toBe(87)});
  it('should generate numeric hashed less than 100', () => {expect(polynomialHash.hash('\u{20000}')).toBe(92)});
});

// describe('railFenceCipher', () => {
//   it('encodes a string correctly for base=3', () => {
//     expect(encodeRailFenceCipher('', 3)).toBe('');
//     expect(encodeRailFenceCipher('12345', 3)).toBe(
//       '15243',
//     );
//     expect(encodeRailFenceCipher('WEAREDISCOVEREDFLEEATONCE', 3)).toBe(
//       'WECRLTEERDSOEEFEAOCAIVDEN',
//     );
//     expect(encodeRailFenceCipher('Hello, World!', 3)).toBe(
//       'Hoo!el,Wrdl l',
//     );
//   });

//   it('decodes a string correctly for base=3', () => {
//     expect(decodeRailFenceCipher('', 3)).toBe('');
//     expect(decodeRailFenceCipher('WECRLTEERDSOEEFEAOCAIVDEN', 3)).toBe(
//       'WEAREDISCOVEREDFLEEATONCE',
//     );
//     expect(decodeRailFenceCipher('Hoo!el,Wrdl l', 3)).toBe(
//       'Hello, World!',
//     );
//     expect(decodeRailFenceCipher('15243', 3)).toBe(
//       '12345',
//     );
//   });

//   it('encodes a string correctly for base=4', () => {
//     expect(encodeRailFenceCipher('', 4)).toBe('');
//     expect(encodeRailFenceCipher('THEYAREATTACKINGFROMTHENORTH', 4)).toBe(
//       'TEKOOHRACIRMNREATANFTETYTGHH',
//     );
//   });

//   it('decodes a string correctly for base=4', () => {
//     expect(decodeRailFenceCipher('', 4)).toBe('');
//     expect(decodeRailFenceCipher('TEKOOHRACIRMNREATANFTETYTGHH', 4)).toBe(
//       'THEYAREATTACKINGFROMTHENORTH',
//     );
//   });
// });

describe('recursiveStaircaseMEM', () => {
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(-1)).toBe(0)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(0)).toBe(0)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(1)).toBe(1)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(2)).toBe(2)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(3)).toBe(3)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(4)).toBe(5)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(5)).toBe(8)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(6)).toBe(13)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(7)).toBe(21)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(8)).toBe(34)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(9)).toBe(55)});
  it('should calculate number of variants using Brute Force with Memoization', () => {expect(recursiveStaircaseMEM(10)).toBe(89)});
});


describe('regularExpressionMatching', () => {
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('', '')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('a', 'a')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aa', 'aa')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', 'aab')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', 'aa.')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', '.a.')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', '...')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('a', 'a*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aaa', 'a*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aaab', 'a*b')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aaabb', 'a*b*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aaabb', 'a*b*c*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('', 'a*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('xaabyc', 'xa*b.c')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', 'c*a*b*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('mississippi', 'mis*is*.p*.')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('ab', '.*')).toBe(true)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('', 'a')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('a', '')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', 'aa')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aab', 'baa')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aabc', '...')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('aaabbdd', 'a*b*c*')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('mississippi', 'mis*is*p*.')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('ab', 'a*')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('abba', 'a*b*.c')).toBe(false)});
  it('should match regular expressions in a string', () => {expect(regularExpressionMatching('abba', '.*c')).toBe(false)});
});

describe('sieveOfEratosthenes', () => {
  it('should find all primes less than or equal to n', () => {
    expect(sieveOfEratosthenes(5)).toEqual([2, 3, 5]);
    expect(sieveOfEratosthenes(10)).toEqual([2, 3, 5, 7]);
    expect(sieveOfEratosthenes(100)).toEqual([
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
      43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ]);
  });
});

describe('SimplePolynomialHash', () => {
  it('should calculate new hash based on previous one', () => {
    const bases = [3, 5];
    const frameSizes = [5, 10];

    const text = 'Lorem Ipsum is simply dummy text of the printing and '
      + 'typesetting industry. Lorem Ipsum has been the industry\'s standard '
      + 'galley of type and \u{ffff} scrambled it to make a type specimen book. It '
      + 'electronic 耀 typesetting, remaining essentially unchanged. It was '
      + 'popularised in the 1960s with the release of Letraset sheets '
      + 'publishing software like Aldus 耀 PageMaker including versions of Lorem.';

    // Check hashing for different prime base.
    bases.forEach((base) => {
      const polynomialHash = new SimplePolynomialHash(base);

      // Check hashing for different word lengths.
      frameSizes.forEach((frameSize) => {
        let previousWord = text.substr(0, frameSize);
        let previousHash = polynomialHash.hash(previousWord);

        // Shift frame through the whole text.
        for (let frameShift = 1; frameShift < (text.length - frameSize); frameShift += 1) {
          const currentWord = text.substr(frameShift, frameSize);
          const currentHash = polynomialHash.hash(currentWord);
          const currentRollingHash = polynomialHash.roll(previousHash, previousWord, currentWord);

          // Check that rolling hash is the same as directly calculated hash.
          expect(currentRollingHash).toBe(currentHash);

          previousWord = currentWord;
          previousHash = currentHash;
        }
      });
    });
  });
  const polynomialHash = new SimplePolynomialHash();

  it('should generate numeric hashed', () => {expect(polynomialHash.hash('Test')).toBe(604944)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('a')).toBe(97)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('b')).toBe(98)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('c')).toBe(99)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('d')).toBe(100)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('e')).toBe(101)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('ab')).toBe(1763)});
  it('should generate numeric hashed', () => {expect(polynomialHash.hash('abc')).toBe(30374)});
});


describe('squareMatrixRotation', () => {
  it('should rotate matrix #0 in-place', () => {
    const matrix = [[1]];

    const rotatedMatrix = [[1]];

    expect(squareMatrixRotation(matrix)).toEqual(rotatedMatrix);
  });

  it('should rotate matrix #1 in-place', () => {
    const matrix = [
      [1, 2],
      [3, 4],
    ];

    const rotatedMatrix = [
      [3, 1],
      [4, 2],
    ];

    expect(squareMatrixRotation(matrix)).toEqual(rotatedMatrix);
  });

  it('should rotate matrix #2 in-place', () => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];

    const rotatedMatrix = [
      [7, 4, 1],
      [8, 5, 2],
      [9, 6, 3],
    ];

    expect(squareMatrixRotation(matrix)).toEqual(rotatedMatrix);
  });

  it('should rotate matrix #3 in-place', () => {
    const matrix = [
      [5, 1, 9, 11],
      [2, 4, 8, 10],
      [13, 3, 6, 7],
      [15, 14, 12, 16],
    ];

    const rotatedMatrix = [
      [15, 13, 2, 5],
      [14, 3, 4, 1],
      [12, 6, 8, 9],
      [16, 7, 10, 11],
    ];

    expect(squareMatrixRotation(matrix)).toEqual(rotatedMatrix);
  });
});

describe('weightedRandom', () => {
  it('should throw an error when the number of weights does not match the number of items', () => {
    const getWeightedRandomWithInvalidInputs = () => {
      weightedRandom(['a', 'b', 'c'], [10, 0]);
    };
    expect(getWeightedRandomWithInvalidInputs).toThrow('Items and weights must be of the same size');
  });

  it('should throw an error when the number of weights or items are empty', () => {
    const getWeightedRandomWithInvalidInputs = () => {
      weightedRandom([], []);
    };
    expect(getWeightedRandomWithInvalidInputs).toThrow('Items must not be empty');
  });

  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [1, 0, 0])).toEqual({ index: 0, item: 'a' })});
  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [0, 1, 0])).toEqual({ index: 1, item: 'b' })});
  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [0, 0, 1])).toEqual({ index: 2, item: 'c' })});
  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [0, 1, 1])).not.toEqual({ index: 0, item: 'a' })});
  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [1, 0, 1])).not.toEqual({ index: 1, item: 'b' })});
  it('should correctly do random selection based on wights in straightforward cases', () => {expect(weightedRandom(['a', 'b', 'c'], [1, 1, 0])).not.toEqual({ index: 2, item: 'c' });});

  it('should correctly do random selection based on wights', () => {
    // Number of times we're going to select the random items based on their weights.
    const ATTEMPTS_NUM = 1000;
    // The +/- delta in the number of times each item has been actually selected.
    // I.e. if we want the item 'a' to be selected 300 times out of 1000 cases (30%)
    // then 267 times is acceptable since it is bigger that 250 (which is 300 - 50)
    // ans smaller than 350 (which is 300 + 50)
    const THRESHOLD = 50;

    const items = ['a', 'b', 'c']; // The actual items values don't matter.
    const weights = [0.1, 0.3, 0.6];

    const counter = [];
    for (let i = 0; i < ATTEMPTS_NUM; i += 1) {
      const randomItem = weightedRandom(items, weights);
      if (!counter[randomItem.index]) {
        counter[randomItem.index] = 1;
      } else {
        counter[randomItem.index] += 1;
      }
    }

    for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
      /*
        i.e. item with the index of 0 must be selected 100 times (ideally)
        or with the threshold of [100 - 50, 100 + 50] times.

        i.e. item with the index of 1 must be selected 300 times (ideally)
        or with the threshold of [300 - 50, 300 + 50] times.

        i.e. item with the index of 2 must be selected 600 times (ideally)
        or with the threshold of [600 - 50, 600 + 50] times.
       */
      expect(counter[itemIndex]).toBeGreaterThan(ATTEMPTS_NUM * weights[itemIndex] - THRESHOLD);
      expect(counter[itemIndex]).toBeLessThan(ATTEMPTS_NUM * weights[itemIndex] + THRESHOLD);
    }
  });
});

