import zAlgorithm from '../../../Before_Refactor/javascript-algorithms/zAlgorithm'
import { caesarCipherEncrypt, caesarCipherDecrypt } from '../../../Before_Refactor/javascript-algorithms/caesarCipher'
import dpRainTerraces from '../../../Before_Refactor/javascript-algorithms/dpRainTerraces'
import dqBestTimeToBuySellStocks from '../../../Before_Refactor/javascript-algorithms/dqBestTimeToBuySellStocks'
import euclideanAlgorithm from '../../../Before_Refactor/javascript-algorithms/euclideanAlgorithm';
import fastPowering from '../../../Before_Refactor/javascript-algorithms/fastPowering';
import greedyJumpGame from '../../../Before_Refactor/javascript-algorithms/greedyJumpGame';
import isPalindrome from '../../../Before_Refactor/javascript-algorithms/isPalindrome';
import knightTour from '../../../Before_Refactor/javascript-algorithms/knightTour';
import knuthMorrisPratt from '../../../Before_Refactor/javascript-algorithms/knuthMorrisPratt';
import levenshteinDistance from '../../../Before_Refactor/javascript-algorithms/levenshteinDistance';
import PolynomialHash from '../../../Before_Refactor/javascript-algorithms/PolynomialHash';
import { encodeRailFenceCipher, decodeRailFenceCipher } from '../../../Before_Refactor/javascript-algorithms/railFenceCipher';
import recursiveStaircaseMEM from '../../../Before_Refactor/javascript-algorithms/recursiveStaircaseMEM';
import regularExpressionMatching from '../../../Before_Refactor/javascript-algorithms/regularExpressionMatching';
import sieveOfEratosthenes from '../../../Before_Refactor/javascript-algorithms/sieveOfEratosthenes';
import SimplePolynomialHash from '../../../Before_Refactor/javascript-algorithms/SimplePolynomialHash';
import squareMatrixRotation from '../../../Before_Refactor/javascript-algorithms/squareMatrixRotation';
import weightedRandom from '../../../Before_Refactor/javascript-algorithms/weightedRandom';

import articulationPoints from '../../../Before_Refactor/javascript-algorithms/articulationPoints'
import bellmanFord from '../../../Before_Refactor/javascript-algorithms/bellmanFord'
import bfTravellingSalesman from '../../../Before_Refactor/javascript-algorithms/bfTravellingSalesman';
import binarySearch from '../../../Before_Refactor/javascript-algorithms/binarySearch'
import breadthFirstSearch from '../../../Before_Refactor/javascript-algorithms/breadthFirstSearch';
import BubbleSort from '../../../Before_Refactor/javascript-algorithms/BubbleSort';
import depthFirstSearch from '../../../Before_Refactor/javascript-algorithms/depthFirstSearch';
import detectDirectedCycle from '../../../Before_Refactor/javascript-algorithms/detectDirectedCycle';
import dijkstra from '../../../Before_Refactor/javascript-algorithms/dijkstra';
import eulerianPath from '../../../Before_Refactor/javascript-algorithms/eulerianPath';
import fastFourierTransform from '../../../Before_Refactor/javascript-algorithms/fastFourierTransform';
import floydWarshall from '../../../Before_Refactor/javascript-algorithms/floydWarshall';
import graphBridges from '../../../Before_Refactor/javascript-algorithms/graphBridges';

import {
  equalArr,
  notSortedArr,
  reverseArr,
  sortedArr,
  SortTester,
} from '../../../CONSTANT/javascript-algorithms/SortTester';
import Graph from '../../../CONSTANT/javascript-algorithms/Graph'
import GraphEdge from '../../../CONSTANT/javascript-algorithms/GraphEdge'
import GraphVertex from '../../../CONSTANT/javascript-algorithms/GraphVertex'
import ComplexNumber from '../../../CONSTANT/javascript-algorithms/ComplexNumber';

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

describe('railFenceCipher', () => {
  it('encodes a string correctly for base=3', () => {
    expect(encodeRailFenceCipher('', 3)).toBe('');
    expect(encodeRailFenceCipher('12345', 3)).toBe(
      '15243',
    );
    expect(encodeRailFenceCipher('WEAREDISCOVEREDFLEEATONCE', 3)).toBe(
      'WECRLTEERDSOEEFEAOCAIVDEN',
    );
    expect(encodeRailFenceCipher('Hello, World!', 3)).toBe(
      'Hoo!el,Wrdl l',
    );
  });

  it('decodes a string correctly for base=3', () => {
    expect(decodeRailFenceCipher('', 3)).toBe('');
    expect(decodeRailFenceCipher('WECRLTEERDSOEEFEAOCAIVDEN', 3)).toBe(
      'WEAREDISCOVEREDFLEEATONCE',
    );
    expect(decodeRailFenceCipher('Hoo!el,Wrdl l', 3)).toBe(
      'Hello, World!',
    );
    expect(decodeRailFenceCipher('15243', 3)).toBe(
      '12345',
    );
  });

  it('encodes a string correctly for base=4', () => {
    expect(encodeRailFenceCipher('', 4)).toBe('');
    expect(encodeRailFenceCipher('THEYAREATTACKINGFROMTHENORTH', 4)).toBe(
      'TEKOOHRACIRMNREATANFTETYTGHH',
    );
  });

  it('decodes a string correctly for base=4', () => {
    expect(decodeRailFenceCipher('', 4)).toBe('');
    expect(decodeRailFenceCipher('TEKOOHRACIRMNREATANFTETYTGHH', 4)).toBe(
      'THEYAREATTACKINGFROMTHENORTH',
    );
  });
});

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


describe('articulationPoints', () => {
  it('should find articulation points in simple graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCD);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(2);
    expect(articulationPointsSet[0].getKey()).toBe(vertexC.getKey());
    expect(articulationPointsSet[1].getKey()).toBe(vertexB.getKey());
  });

  it('should find articulation points in simple graph with back edge', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeAC = new GraphEdge(vertexA, vertexC);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(1);
    expect(articulationPointsSet[0].getKey()).toBe(vertexC.getKey());
  });

  it('should find articulation points in simple graph with back edge #2', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeCE = new GraphEdge(vertexC, vertexE);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAE)
      .addEdge(edgeCE)
      .addEdge(edgeBC)
      .addEdge(edgeCD);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(1);
    expect(articulationPointsSet[0].getKey()).toBe(vertexC.getKey());
  });

  it('should find articulation points in graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeGF = new GraphEdge(vertexG, vertexF);
    const edgeFH = new GraphEdge(vertexF, vertexH);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeAC)
      .addEdge(edgeCD)
      .addEdge(edgeDE)
      .addEdge(edgeEG)
      .addEdge(edgeEF)
      .addEdge(edgeGF)
      .addEdge(edgeFH);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(4);
    expect(articulationPointsSet[0].getKey()).toBe(vertexF.getKey());
    expect(articulationPointsSet[1].getKey()).toBe(vertexE.getKey());
    expect(articulationPointsSet[2].getKey()).toBe(vertexD.getKey());
    expect(articulationPointsSet[3].getKey()).toBe(vertexC.getKey());
  });

  it('should find articulation points in graph starting with articulation root vertex', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeGF = new GraphEdge(vertexG, vertexF);
    const edgeFH = new GraphEdge(vertexF, vertexH);

    const graph = new Graph();

    graph
      .addEdge(edgeDE)
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeAC)
      .addEdge(edgeCD)
      .addEdge(edgeEG)
      .addEdge(edgeEF)
      .addEdge(edgeGF)
      .addEdge(edgeFH);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(4);
    expect(articulationPointsSet[0].getKey()).toBe(vertexF.getKey());
    expect(articulationPointsSet[1].getKey()).toBe(vertexE.getKey());
    expect(articulationPointsSet[2].getKey()).toBe(vertexC.getKey());
    expect(articulationPointsSet[3].getKey()).toBe(vertexD.getKey());
  });

  it('should find articulation points in yet another graph #1', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD)
      .addEdge(edgeDE);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(2);
    expect(articulationPointsSet[0].getKey()).toBe(vertexD.getKey());
    expect(articulationPointsSet[1].getKey()).toBe(vertexC.getKey());
  });

  it('should find articulation points in yet another graph #2', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeCE = new GraphEdge(vertexC, vertexE);
    const edgeCF = new GraphEdge(vertexC, vertexF);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeFG = new GraphEdge(vertexF, vertexG);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD)
      .addEdge(edgeCE)
      .addEdge(edgeCF)
      .addEdge(edgeEG)
      .addEdge(edgeFG);

    const articulationPointsSet = Object.values(articulationPoints(graph));

    expect(articulationPointsSet.length).toBe(1);
    expect(articulationPointsSet[0].getKey()).toBe(vertexC.getKey());
  });
});

describe('bellmanFord', () => {
  it('should find minimum paths to all vertices for undirected graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB, 4);
    const edgeAE = new GraphEdge(vertexA, vertexE, 7);
    const edgeAC = new GraphEdge(vertexA, vertexC, 3);
    const edgeBC = new GraphEdge(vertexB, vertexC, 6);
    const edgeBD = new GraphEdge(vertexB, vertexD, 5);
    const edgeEC = new GraphEdge(vertexE, vertexC, 8);
    const edgeED = new GraphEdge(vertexE, vertexD, 2);
    const edgeDC = new GraphEdge(vertexD, vertexC, 11);
    const edgeDG = new GraphEdge(vertexD, vertexG, 10);
    const edgeDF = new GraphEdge(vertexD, vertexF, 2);
    const edgeFG = new GraphEdge(vertexF, vertexG, 3);
    const edgeEG = new GraphEdge(vertexE, vertexG, 5);

    const graph = new Graph();
    graph
      .addVertex(vertexH)
      .addEdge(edgeAB)
      .addEdge(edgeAE)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeBD)
      .addEdge(edgeEC)
      .addEdge(edgeED)
      .addEdge(edgeDC)
      .addEdge(edgeDG)
      .addEdge(edgeDF)
      .addEdge(edgeFG)
      .addEdge(edgeEG);

    const { distances, previousVertices } = bellmanFord(graph, vertexA);

    expect(distances).toEqual({
      H: Infinity,
      A: 0,
      B: 4,
      E: 7,
      C: 3,
      D: 9,
      G: 12,
      F: 11,
    });

    expect(previousVertices.F.getKey()).toBe('D');
    expect(previousVertices.D.getKey()).toBe('B');
    expect(previousVertices.B.getKey()).toBe('A');
    expect(previousVertices.G.getKey()).toBe('E');
    expect(previousVertices.C.getKey()).toBe('A');
    expect(previousVertices.A).toBeNull();
    expect(previousVertices.H).toBeNull();
  });

  it('should find minimum paths to all vertices for directed graph with negative edge weights', () => {
    const vertexS = new GraphVertex('S');
    const vertexE = new GraphVertex('E');
    const vertexA = new GraphVertex('A');
    const vertexD = new GraphVertex('D');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexH = new GraphVertex('H');

    const edgeSE = new GraphEdge(vertexS, vertexE, 8);
    const edgeSA = new GraphEdge(vertexS, vertexA, 10);
    const edgeED = new GraphEdge(vertexE, vertexD, 1);
    const edgeDA = new GraphEdge(vertexD, vertexA, -4);
    const edgeDC = new GraphEdge(vertexD, vertexC, -1);
    const edgeAC = new GraphEdge(vertexA, vertexC, 2);
    const edgeCB = new GraphEdge(vertexC, vertexB, -2);
    const edgeBA = new GraphEdge(vertexB, vertexA, 1);

    const graph = new Graph(true);
    graph
      .addVertex(vertexH)
      .addEdge(edgeSE)
      .addEdge(edgeSA)
      .addEdge(edgeED)
      .addEdge(edgeDA)
      .addEdge(edgeDC)
      .addEdge(edgeAC)
      .addEdge(edgeCB)
      .addEdge(edgeBA);

    const { distances, previousVertices } = bellmanFord(graph, vertexS);

    expect(distances).toEqual({
      H: Infinity,
      S: 0,
      A: 5,
      B: 5,
      C: 7,
      D: 9,
      E: 8,
    });

    expect(previousVertices.H).toBeNull();
    expect(previousVertices.S).toBeNull();
    expect(previousVertices.B.getKey()).toBe('C');
    expect(previousVertices.C.getKey()).toBe('A');
    expect(previousVertices.A.getKey()).toBe('D');
    expect(previousVertices.D.getKey()).toBe('E');
  });
});


describe('bfTravellingSalesman', () => {
  it('should solve problem for simple graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB, 1);
    const edgeBD = new GraphEdge(vertexB, vertexD, 1);
    const edgeDC = new GraphEdge(vertexD, vertexC, 1);
    const edgeCA = new GraphEdge(vertexC, vertexA, 1);

    const edgeBA = new GraphEdge(vertexB, vertexA, 5);
    const edgeDB = new GraphEdge(vertexD, vertexB, 8);
    const edgeCD = new GraphEdge(vertexC, vertexD, 7);
    const edgeAC = new GraphEdge(vertexA, vertexC, 4);
    const edgeAD = new GraphEdge(vertexA, vertexD, 2);
    const edgeDA = new GraphEdge(vertexD, vertexA, 3);
    const edgeBC = new GraphEdge(vertexB, vertexC, 3);
    const edgeCB = new GraphEdge(vertexC, vertexB, 9);

    const graph = new Graph(true);
    graph
      .addEdge(edgeAB)
      .addEdge(edgeBD)
      .addEdge(edgeDC)
      .addEdge(edgeCA)
      .addEdge(edgeBA)
      .addEdge(edgeDB)
      .addEdge(edgeCD)
      .addEdge(edgeAC)
      .addEdge(edgeAD)
      .addEdge(edgeDA)
      .addEdge(edgeBC)
      .addEdge(edgeCB);

    const salesmanPath = bfTravellingSalesman(graph);

    expect(salesmanPath.length).toBe(4);

    expect(salesmanPath[0].getKey()).toEqual(vertexA.getKey());
    expect(salesmanPath[1].getKey()).toEqual(vertexB.getKey());
    expect(salesmanPath[2].getKey()).toEqual(vertexD.getKey());
    expect(salesmanPath[3].getKey()).toEqual(vertexC.getKey());
  });
});

describe('binarySearch', () => {
  it('should search number in sorted array', () => {
    expect(binarySearch([], 1)).toBe(-1);
    expect(binarySearch([1], 1)).toBe(0);
    expect(binarySearch([1, 2], 1)).toBe(0);
    expect(binarySearch([1, 2], 2)).toBe(1);
    expect(binarySearch([1, 5, 10, 12], 1)).toBe(0);
    expect(binarySearch([1, 5, 10, 12, 14, 17, 22, 100], 17)).toBe(5);
    expect(binarySearch([1, 5, 10, 12, 14, 17, 22, 100], 1)).toBe(0);
    expect(binarySearch([1, 5, 10, 12, 14, 17, 22, 100], 100)).toBe(7);
    expect(binarySearch([1, 5, 10, 12, 14, 17, 22, 100], 0)).toBe(-1);
  });

  it('should search object in sorted array', () => {
    const sortedArrayOfObjects = [
      { key: 1, value: 'value1' },
      { key: 2, value: 'value2' },
      { key: 3, value: 'value3' },
    ];

    const comparator = (a, b) => {
      if (a.key === b.key) return 0;
      return a.key < b.key ? -1 : 1;
    };

    expect(binarySearch([], { key: 1 }, comparator)).toBe(-1);
    expect(binarySearch(sortedArrayOfObjects, { key: 4 }, comparator)).toBe(-1);
    expect(binarySearch(sortedArrayOfObjects, { key: 1 }, comparator)).toBe(0);
    expect(binarySearch(sortedArrayOfObjects, { key: 2 }, comparator)).toBe(1);
    expect(binarySearch(sortedArrayOfObjects, { key: 3 }, comparator)).toBe(2);
  });
});

describe('breadthFirstSearch', () => {
  it('should perform BFS operation on graph', () => {
    const graph = new Graph(true);

    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCG = new GraphEdge(vertexC, vertexG);
    const edgeAD = new GraphEdge(vertexA, vertexD);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFD = new GraphEdge(vertexF, vertexD);
    const edgeDH = new GraphEdge(vertexD, vertexH);
    const edgeGH = new GraphEdge(vertexG, vertexH);

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCG)
      .addEdge(edgeAD)
      .addEdge(edgeAE)
      .addEdge(edgeEF)
      .addEdge(edgeFD)
      .addEdge(edgeDH)
      .addEdge(edgeGH);

    expect(graph.toString()).toBe('A,B,C,G,D,E,F,H');

    const enterVertexCallback = jest.fn();
    const leaveVertexCallback = jest.fn();

    // Traverse graphs without callbacks first.
    breadthFirstSearch(graph, vertexA);

    // Traverse graph with enterVertex and leaveVertex callbacks.
    breadthFirstSearch(graph, vertexA, {
      enterVertex: enterVertexCallback,
      leaveVertex: leaveVertexCallback,
    });

    expect(enterVertexCallback).toHaveBeenCalledTimes(8);
    expect(leaveVertexCallback).toHaveBeenCalledTimes(8);

    const enterVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexB, previousVertex: vertexA },
      { currentVertex: vertexD, previousVertex: vertexB },
      { currentVertex: vertexE, previousVertex: vertexD },
      { currentVertex: vertexC, previousVertex: vertexE },
      { currentVertex: vertexH, previousVertex: vertexC },
      { currentVertex: vertexF, previousVertex: vertexH },
      { currentVertex: vertexG, previousVertex: vertexF },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = enterVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(enterVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(enterVertexParamsMap[callIndex].previousVertex);
    }

    const leaveVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexB, previousVertex: vertexA },
      { currentVertex: vertexD, previousVertex: vertexB },
      { currentVertex: vertexE, previousVertex: vertexD },
      { currentVertex: vertexC, previousVertex: vertexE },
      { currentVertex: vertexH, previousVertex: vertexC },
      { currentVertex: vertexF, previousVertex: vertexH },
      { currentVertex: vertexG, previousVertex: vertexF },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = leaveVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(leaveVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(leaveVertexParamsMap[callIndex].previousVertex);
    }
  });

  it('should allow to create custom vertex visiting logic', () => {
    const graph = new Graph(true);

    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCG = new GraphEdge(vertexC, vertexG);
    const edgeAD = new GraphEdge(vertexA, vertexD);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFD = new GraphEdge(vertexF, vertexD);
    const edgeDH = new GraphEdge(vertexD, vertexH);
    const edgeGH = new GraphEdge(vertexG, vertexH);

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCG)
      .addEdge(edgeAD)
      .addEdge(edgeAE)
      .addEdge(edgeEF)
      .addEdge(edgeFD)
      .addEdge(edgeDH)
      .addEdge(edgeGH);

    expect(graph.toString()).toBe('A,B,C,G,D,E,F,H');

    const enterVertexCallback = jest.fn();
    const leaveVertexCallback = jest.fn();

    // Traverse graph with enterVertex and leaveVertex callbacks.
    breadthFirstSearch(graph, vertexA, {
      enterVertex: enterVertexCallback,
      leaveVertex: leaveVertexCallback,
      allowTraversal: ({ currentVertex, nextVertex }) => {
        return !(currentVertex === vertexA && nextVertex === vertexB);
      },
    });

    expect(enterVertexCallback).toHaveBeenCalledTimes(7);
    expect(leaveVertexCallback).toHaveBeenCalledTimes(7);

    const enterVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexE, previousVertex: vertexD },
      { currentVertex: vertexH, previousVertex: vertexE },
      { currentVertex: vertexF, previousVertex: vertexH },
      { currentVertex: vertexD, previousVertex: vertexF },
      { currentVertex: vertexH, previousVertex: vertexD },
    ];

    for (let callIndex = 0; callIndex < 7; callIndex += 1) {
      const params = enterVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(enterVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(enterVertexParamsMap[callIndex].previousVertex);
    }

    const leaveVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexE, previousVertex: vertexD },
      { currentVertex: vertexH, previousVertex: vertexE },
      { currentVertex: vertexF, previousVertex: vertexH },
      { currentVertex: vertexD, previousVertex: vertexF },
      { currentVertex: vertexH, previousVertex: vertexD },
    ];

    for (let callIndex = 0; callIndex < 7; callIndex += 1) {
      const params = leaveVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(leaveVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(leaveVertexParamsMap[callIndex].previousVertex);
    }
  });
});


// Complexity constants.
const SORTED_ARRAY_VISITING_COUNT = 20;
const NOT_SORTED_ARRAY_VISITING_COUNT = 189;
const REVERSE_SORTED_ARRAY_VISITING_COUNT = 209;
const EQUAL_ARRAY_VISITING_COUNT = 20;

describe('BubbleSort', () => {
  it('should sort array', () => {
    SortTester.testSort(BubbleSort);
  });

  it('should sort array with custom comparator', () => {
    SortTester.testSortWithCustomComparator(BubbleSort);
  });

  it('should do stable sorting', () => {
    SortTester.testSortStability(BubbleSort);
  });

  it('should sort negative numbers', () => {
    SortTester.testNegativeNumbersSort(BubbleSort);
  });

  it('should visit EQUAL array element specified number of times', () => {
    SortTester.testAlgorithmTimeComplexity(
      BubbleSort,
      equalArr,
      EQUAL_ARRAY_VISITING_COUNT,
    );
  });

  it('should visit SORTED array element specified number of times', () => {
    SortTester.testAlgorithmTimeComplexity(
      BubbleSort,
      sortedArr,
      SORTED_ARRAY_VISITING_COUNT,
    );
  });

  it('should visit NOT SORTED array element specified number of times', () => {
    SortTester.testAlgorithmTimeComplexity(
      BubbleSort,
      notSortedArr,
      NOT_SORTED_ARRAY_VISITING_COUNT,
    );
  });

  it('should visit REVERSE SORTED array element specified number of times', () => {
    SortTester.testAlgorithmTimeComplexity(
      BubbleSort,
      reverseArr,
      REVERSE_SORTED_ARRAY_VISITING_COUNT,
    );
  });
});

describe('depthFirstSearch', () => {
  it('should perform DFS operation on graph', () => {
    const graph = new Graph(true);

    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCG = new GraphEdge(vertexC, vertexG);
    const edgeAD = new GraphEdge(vertexA, vertexD);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFD = new GraphEdge(vertexF, vertexD);
    const edgeDG = new GraphEdge(vertexD, vertexG);

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCG)
      .addEdge(edgeAD)
      .addEdge(edgeAE)
      .addEdge(edgeEF)
      .addEdge(edgeFD)
      .addEdge(edgeDG);

    expect(graph.toString()).toBe('A,B,C,G,D,E,F');

    const enterVertexCallback = jest.fn();
    const leaveVertexCallback = jest.fn();

    // Traverse graphs without callbacks first to check default ones.
    depthFirstSearch(graph, vertexA);

    // Traverse graph with enterVertex and leaveVertex callbacks.
    depthFirstSearch(graph, vertexA, {
      enterVertex: enterVertexCallback,
      leaveVertex: leaveVertexCallback,
    });

    expect(enterVertexCallback).toHaveBeenCalledTimes(graph.getAllVertices().length);
    expect(leaveVertexCallback).toHaveBeenCalledTimes(graph.getAllVertices().length);

    const enterVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexB, previousVertex: vertexA },
      { currentVertex: vertexC, previousVertex: vertexB },
      { currentVertex: vertexG, previousVertex: vertexC },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexE, previousVertex: vertexA },
      { currentVertex: vertexF, previousVertex: vertexE },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = enterVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(enterVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(enterVertexParamsMap[callIndex].previousVertex);
    }

    const leaveVertexParamsMap = [
      { currentVertex: vertexG, previousVertex: vertexC },
      { currentVertex: vertexC, previousVertex: vertexB },
      { currentVertex: vertexB, previousVertex: vertexA },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexF, previousVertex: vertexE },
      { currentVertex: vertexE, previousVertex: vertexA },
      { currentVertex: vertexA, previousVertex: null },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = leaveVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(leaveVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(leaveVertexParamsMap[callIndex].previousVertex);
    }
  });

  it('allow users to redefine vertex visiting logic', () => {
    const graph = new Graph(true);

    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCG = new GraphEdge(vertexC, vertexG);
    const edgeAD = new GraphEdge(vertexA, vertexD);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFD = new GraphEdge(vertexF, vertexD);
    const edgeDG = new GraphEdge(vertexD, vertexG);

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCG)
      .addEdge(edgeAD)
      .addEdge(edgeAE)
      .addEdge(edgeEF)
      .addEdge(edgeFD)
      .addEdge(edgeDG);

    expect(graph.toString()).toBe('A,B,C,G,D,E,F');

    const enterVertexCallback = jest.fn();
    const leaveVertexCallback = jest.fn();

    depthFirstSearch(graph, vertexA, {
      enterVertex: enterVertexCallback,
      leaveVertex: leaveVertexCallback,
      allowTraversal: ({ currentVertex, nextVertex }) => {
        return !(currentVertex === vertexA && nextVertex === vertexB);
      },
    });

    expect(enterVertexCallback).toHaveBeenCalledTimes(7);
    expect(leaveVertexCallback).toHaveBeenCalledTimes(7);

    const enterVertexParamsMap = [
      { currentVertex: vertexA, previousVertex: null },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexG, previousVertex: vertexD },
      { currentVertex: vertexE, previousVertex: vertexA },
      { currentVertex: vertexF, previousVertex: vertexE },
      { currentVertex: vertexD, previousVertex: vertexF },
      { currentVertex: vertexG, previousVertex: vertexD },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = enterVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(enterVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(enterVertexParamsMap[callIndex].previousVertex);
    }

    const leaveVertexParamsMap = [
      { currentVertex: vertexG, previousVertex: vertexD },
      { currentVertex: vertexD, previousVertex: vertexA },
      { currentVertex: vertexG, previousVertex: vertexD },
      { currentVertex: vertexD, previousVertex: vertexF },
      { currentVertex: vertexF, previousVertex: vertexE },
      { currentVertex: vertexE, previousVertex: vertexA },
      { currentVertex: vertexA, previousVertex: null },
    ];

    for (let callIndex = 0; callIndex < graph.getAllVertices().length; callIndex += 1) {
      const params = leaveVertexCallback.mock.calls[callIndex][0];
      expect(params.currentVertex).toEqual(leaveVertexParamsMap[callIndex].currentVertex);
      expect(params.previousVertex).toEqual(leaveVertexParamsMap[callIndex].previousVertex);
    }
  });
});

describe('detectDirectedCycle', () => {
  it('should detect directed cycle', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeDA = new GraphEdge(vertexD, vertexA);
    const edgeDE = new GraphEdge(vertexD, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFD = new GraphEdge(vertexF, vertexD);

    const graph = new Graph(true);
    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeAC)
      .addEdge(edgeDA)
      .addEdge(edgeDE)
      .addEdge(edgeEF);

    expect(detectDirectedCycle(graph)).toBeNull();

    graph.addEdge(edgeFD);

    expect(detectDirectedCycle(graph)).toEqual({
      D: vertexF,
      F: vertexE,
      E: vertexD,
    });
  });
});

describe('dijkstra', () => {
  it('should find minimum paths to all vertices for undirected graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB, 4);
    const edgeAE = new GraphEdge(vertexA, vertexE, 7);
    const edgeAC = new GraphEdge(vertexA, vertexC, 3);
    const edgeBC = new GraphEdge(vertexB, vertexC, 6);
    const edgeBD = new GraphEdge(vertexB, vertexD, 5);
    const edgeEC = new GraphEdge(vertexE, vertexC, 8);
    const edgeED = new GraphEdge(vertexE, vertexD, 2);
    const edgeDC = new GraphEdge(vertexD, vertexC, 11);
    const edgeDG = new GraphEdge(vertexD, vertexG, 10);
    const edgeDF = new GraphEdge(vertexD, vertexF, 2);
    const edgeFG = new GraphEdge(vertexF, vertexG, 3);
    const edgeEG = new GraphEdge(vertexE, vertexG, 5);

    const graph = new Graph();
    graph
      .addVertex(vertexH)
      .addEdge(edgeAB)
      .addEdge(edgeAE)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeBD)
      .addEdge(edgeEC)
      .addEdge(edgeED)
      .addEdge(edgeDC)
      .addEdge(edgeDG)
      .addEdge(edgeDF)
      .addEdge(edgeFG)
      .addEdge(edgeEG);

    const { distances, previousVertices } = dijkstra(graph, vertexA);

    expect(distances).toEqual({
      H: Infinity,
      A: 0,
      B: 4,
      E: 7,
      C: 3,
      D: 9,
      G: 12,
      F: 11,
    });

    expect(previousVertices.F.getKey()).toBe('D');
    expect(previousVertices.D.getKey()).toBe('B');
    expect(previousVertices.B.getKey()).toBe('A');
    expect(previousVertices.G.getKey()).toBe('E');
    expect(previousVertices.C.getKey()).toBe('A');
    expect(previousVertices.A).toBeNull();
    expect(previousVertices.H).toBeNull();
  });

  it('should find minimum paths to all vertices for directed graph with negative edge weights', () => {
    const vertexS = new GraphVertex('S');
    const vertexE = new GraphVertex('E');
    const vertexA = new GraphVertex('A');
    const vertexD = new GraphVertex('D');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexH = new GraphVertex('H');

    const edgeSE = new GraphEdge(vertexS, vertexE, 8);
    const edgeSA = new GraphEdge(vertexS, vertexA, 10);
    const edgeED = new GraphEdge(vertexE, vertexD, 1);
    const edgeDA = new GraphEdge(vertexD, vertexA, -4);
    const edgeDC = new GraphEdge(vertexD, vertexC, -1);
    const edgeAC = new GraphEdge(vertexA, vertexC, 2);
    const edgeCB = new GraphEdge(vertexC, vertexB, -2);
    const edgeBA = new GraphEdge(vertexB, vertexA, 1);

    const graph = new Graph(true);
    graph
      .addVertex(vertexH)
      .addEdge(edgeSE)
      .addEdge(edgeSA)
      .addEdge(edgeED)
      .addEdge(edgeDA)
      .addEdge(edgeDC)
      .addEdge(edgeAC)
      .addEdge(edgeCB)
      .addEdge(edgeBA);

    const { distances, previousVertices } = dijkstra(graph, vertexS);

    expect(distances).toEqual({
      H: Infinity,
      S: 0,
      A: 5,
      B: 5,
      C: 7,
      D: 9,
      E: 8,
    });

    expect(previousVertices.H).toBeNull();
    expect(previousVertices.S).toBeNull();
    expect(previousVertices.B.getKey()).toBe('C');
    expect(previousVertices.C.getKey()).toBe('A');
    expect(previousVertices.A.getKey()).toBe('D');
    expect(previousVertices.D.getKey()).toBe('E');
  });
});

describe('eulerianPath', () => {
  it('should throw an error when graph is not Eulerian', () => {
    function findEulerianPathInNotEulerianGraph() {
      const vertexA = new GraphVertex('A');
      const vertexB = new GraphVertex('B');
      const vertexC = new GraphVertex('C');
      const vertexD = new GraphVertex('D');
      const vertexE = new GraphVertex('E');

      const edgeAB = new GraphEdge(vertexA, vertexB);
      const edgeAC = new GraphEdge(vertexA, vertexC);
      const edgeBC = new GraphEdge(vertexB, vertexC);
      const edgeBD = new GraphEdge(vertexB, vertexD);
      const edgeCE = new GraphEdge(vertexC, vertexE);

      const graph = new Graph();

      graph
        .addEdge(edgeAB)
        .addEdge(edgeAC)
        .addEdge(edgeBC)
        .addEdge(edgeBD)
        .addEdge(edgeCE);

      eulerianPath(graph);
    }

    expect(findEulerianPathInNotEulerianGraph).toThrowError();
  });

  it('should find Eulerian Circuit in graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAE = new GraphEdge(vertexA, vertexE);
    const edgeAF = new GraphEdge(vertexA, vertexF);
    const edgeAG = new GraphEdge(vertexA, vertexG);
    const edgeGF = new GraphEdge(vertexG, vertexF);
    const edgeBE = new GraphEdge(vertexB, vertexE);
    const edgeEB = new GraphEdge(vertexE, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeED = new GraphEdge(vertexE, vertexD);
    const edgeCD = new GraphEdge(vertexC, vertexD);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAE)
      .addEdge(edgeAF)
      .addEdge(edgeAG)
      .addEdge(edgeGF)
      .addEdge(edgeBE)
      .addEdge(edgeEB)
      .addEdge(edgeBC)
      .addEdge(edgeED)
      .addEdge(edgeCD);

    const graphEdgesCount = graph.getAllEdges().length;

    const eulerianPathSet = eulerianPath(graph);

    expect(eulerianPathSet.length).toBe(graphEdgesCount + 1);

    expect(eulerianPathSet[0].getKey()).toBe(vertexA.getKey());
    expect(eulerianPathSet[1].getKey()).toBe(vertexB.getKey());
    expect(eulerianPathSet[2].getKey()).toBe(vertexE.getKey());
    expect(eulerianPathSet[3].getKey()).toBe(vertexB.getKey());
    expect(eulerianPathSet[4].getKey()).toBe(vertexC.getKey());
    expect(eulerianPathSet[5].getKey()).toBe(vertexD.getKey());
    expect(eulerianPathSet[6].getKey()).toBe(vertexE.getKey());
    expect(eulerianPathSet[7].getKey()).toBe(vertexA.getKey());
    expect(eulerianPathSet[8].getKey()).toBe(vertexF.getKey());
    expect(eulerianPathSet[9].getKey()).toBe(vertexG.getKey());
    expect(eulerianPathSet[10].getKey()).toBe(vertexA.getKey());
  });

  it('should find Eulerian Path in graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeBD = new GraphEdge(vertexB, vertexD);
    const edgeDC = new GraphEdge(vertexD, vertexC);
    const edgeCE = new GraphEdge(vertexC, vertexE);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeFH = new GraphEdge(vertexF, vertexH);
    const edgeFG = new GraphEdge(vertexF, vertexG);
    const edgeHG = new GraphEdge(vertexH, vertexG);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBD)
      .addEdge(edgeDC)
      .addEdge(edgeCE)
      .addEdge(edgeEF)
      .addEdge(edgeFH)
      .addEdge(edgeFG)
      .addEdge(edgeHG);

    const graphEdgesCount = graph.getAllEdges().length;

    const eulerianPathSet = eulerianPath(graph);

    expect(eulerianPathSet.length).toBe(graphEdgesCount + 1);

    expect(eulerianPathSet[0].getKey()).toBe(vertexC.getKey());
    expect(eulerianPathSet[1].getKey()).toBe(vertexA.getKey());
    expect(eulerianPathSet[2].getKey()).toBe(vertexB.getKey());
    expect(eulerianPathSet[3].getKey()).toBe(vertexD.getKey());
    expect(eulerianPathSet[4].getKey()).toBe(vertexC.getKey());
    expect(eulerianPathSet[5].getKey()).toBe(vertexE.getKey());
    expect(eulerianPathSet[6].getKey()).toBe(vertexF.getKey());
    expect(eulerianPathSet[7].getKey()).toBe(vertexH.getKey());
    expect(eulerianPathSet[8].getKey()).toBe(vertexG.getKey());
    expect(eulerianPathSet[9].getKey()).toBe(vertexF.getKey());
  });
});

/**
 * @param {ComplexNumber[]} sequence1
 * @param {ComplexNumber[]} sequence2
 * @param {Number} delta
 * @return {boolean}
 */
function sequencesApproximatelyEqual(sequence1, sequence2, delta) {
  if (sequence1.length !== sequence2.length) {
    return false;
  }

  for (let numberIndex = 0; numberIndex < sequence1.length; numberIndex += 1) {
    if (Math.abs(sequence1[numberIndex].re - sequence2[numberIndex].re) > delta) {
      return false;
    }

    if (Math.abs(sequence1[numberIndex].im - sequence2[numberIndex].im) > delta) {
      return false;
    }
  }

  return true;
}

const delta = 1e-6;

describe('fastFourierTransform', () => {
  it('should calculate the radix-2 discrete fourier transform #1', () => {
    const input = [new ComplexNumber({ re: 0, im: 0 })];
    const expectedOutput = [new ComplexNumber({ re: 0, im: 0 })];
    const output = fastFourierTransform(input);
    const invertedOutput = fastFourierTransform(output, true);

    expect(sequencesApproximatelyEqual(expectedOutput, output, delta)).toBe(true);
    expect(sequencesApproximatelyEqual(input, invertedOutput, delta)).toBe(true);
  });

  it('should calculate the radix-2 discrete fourier transform #2', () => {
    const input = [
      new ComplexNumber({ re: 1, im: 2 }),
      new ComplexNumber({ re: 2, im: 3 }),
      new ComplexNumber({ re: 8, im: 4 }),
    ];

    const expectedOutput = [
      new ComplexNumber({ re: 11, im: 9 }),
      new ComplexNumber({ re: -10, im: 0 }),
      new ComplexNumber({ re: 7, im: 3 }),
      new ComplexNumber({ re: -4, im: -4 }),
    ];

    const output = fastFourierTransform(input);
    const invertedOutput = fastFourierTransform(output, true);

    expect(sequencesApproximatelyEqual(expectedOutput, output, delta)).toBe(true);
    expect(sequencesApproximatelyEqual(input, invertedOutput, delta)).toBe(true);
  });

  it('should calculate the radix-2 discrete fourier transform #3', () => {
    const input = [
      new ComplexNumber({ re: -83656.9359385182, im: 98724.08038374918 }),
      new ComplexNumber({ re: -47537.415125808424, im: 88441.58381765135 }),
      new ComplexNumber({ re: -24849.657029355192, im: -72621.79007878687 }),
      new ComplexNumber({ re: 31451.27290052717, im: -21113.301128347346 }),
      new ComplexNumber({ re: 13973.90836288876, im: -73378.36721594246 }),
      new ComplexNumber({ re: 14981.520420492234, im: 63279.524958963884 }),
      new ComplexNumber({ re: -9892.575367044381, im: -81748.44671677813 }),
      new ComplexNumber({ re: -35933.00356823792, im: -46153.47157161784 }),
      new ComplexNumber({ re: -22425.008561855735, im: -86284.24507370662 }),
      new ComplexNumber({ re: -39327.43830818355, im: 30611.949874562706 }),
    ];

    const expectedOutput = [
      new ComplexNumber({ re: -203215.3322151, im: -100242.4827503 }),
      new ComplexNumber({ re: 99217.0805705, im: 270646.9331932 }),
      new ComplexNumber({ re: -305990.9040412, im: 68224.8435751 }),
      new ComplexNumber({ re: -14135.7758282, im: 199223.9878095 }),
      new ComplexNumber({ re: -306965.6350922, im: 26030.1025439 }),
      new ComplexNumber({ re: -76477.6755206, im: 40781.9078990 }),
      new ComplexNumber({ re: -48409.3099088, im: 54674.7959662 }),
      new ComplexNumber({ re: -329683.0131713, im: 164287.7995937 }),
      new ComplexNumber({ re: -50485.2048527, im: -330375.0546527 }),
      new ComplexNumber({ re: 122235.7738708, im: 91091.6398019 }),
      new ComplexNumber({ re: 47625.8850387, im: 73497.3981523 }),
      new ComplexNumber({ re: -15619.8231136, im: 80804.8685410 }),
      new ComplexNumber({ re: 192234.0276101, im: 160833.3072355 }),
      new ComplexNumber({ re: -96389.4195635, im: 393408.4543872 }),
      new ComplexNumber({ re: -173449.0825417, im: 146875.7724104 }),
      new ComplexNumber({ re: -179002.5662573, im: 239821.0124341 }),
    ];

    const output = fastFourierTransform(input);
    const invertedOutput = fastFourierTransform(output, true);

    expect(sequencesApproximatelyEqual(expectedOutput, output, delta)).toBe(true);
    expect(sequencesApproximatelyEqual(input, invertedOutput, delta)).toBe(true);
  });
});

describe('floydWarshall', () => {
  it('should find minimum paths to all vertices for undirected graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB, 4);
    const edgeAE = new GraphEdge(vertexA, vertexE, 7);
    const edgeAC = new GraphEdge(vertexA, vertexC, 3);
    const edgeBC = new GraphEdge(vertexB, vertexC, 6);
    const edgeBD = new GraphEdge(vertexB, vertexD, 5);
    const edgeEC = new GraphEdge(vertexE, vertexC, 8);
    const edgeED = new GraphEdge(vertexE, vertexD, 2);
    const edgeDC = new GraphEdge(vertexD, vertexC, 11);
    const edgeDG = new GraphEdge(vertexD, vertexG, 10);
    const edgeDF = new GraphEdge(vertexD, vertexF, 2);
    const edgeFG = new GraphEdge(vertexF, vertexG, 3);
    const edgeEG = new GraphEdge(vertexE, vertexG, 5);

    const graph = new Graph();

    // Add vertices first just to have them in desired order.
    graph
      .addVertex(vertexA)
      .addVertex(vertexB)
      .addVertex(vertexC)
      .addVertex(vertexD)
      .addVertex(vertexE)
      .addVertex(vertexF)
      .addVertex(vertexG)
      .addVertex(vertexH);

    // Now, when vertices are in correct order let's add edges.
    graph
      .addEdge(edgeAB)
      .addEdge(edgeAE)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeBD)
      .addEdge(edgeEC)
      .addEdge(edgeED)
      .addEdge(edgeDC)
      .addEdge(edgeDG)
      .addEdge(edgeDF)
      .addEdge(edgeFG)
      .addEdge(edgeEG);

    const { distances, nextVertices } = floydWarshall(graph);

    const vertices = graph.getAllVertices();

    const vertexAIndex = vertices.indexOf(vertexA);
    const vertexBIndex = vertices.indexOf(vertexB);
    const vertexCIndex = vertices.indexOf(vertexC);
    const vertexDIndex = vertices.indexOf(vertexD);
    const vertexEIndex = vertices.indexOf(vertexE);
    const vertexFIndex = vertices.indexOf(vertexF);
    const vertexGIndex = vertices.indexOf(vertexG);
    const vertexHIndex = vertices.indexOf(vertexH);

    expect(distances[vertexAIndex][vertexHIndex]).toBe(Infinity);
    expect(distances[vertexAIndex][vertexAIndex]).toBe(0);
    expect(distances[vertexAIndex][vertexBIndex]).toBe(4);
    expect(distances[vertexAIndex][vertexEIndex]).toBe(7);
    expect(distances[vertexAIndex][vertexCIndex]).toBe(3);
    expect(distances[vertexAIndex][vertexDIndex]).toBe(9);
    expect(distances[vertexAIndex][vertexGIndex]).toBe(12);
    expect(distances[vertexAIndex][vertexFIndex]).toBe(11);

    expect(nextVertices[vertexAIndex][vertexFIndex]).toBe(vertexD);
    expect(nextVertices[vertexAIndex][vertexDIndex]).toBe(vertexB);
    expect(nextVertices[vertexAIndex][vertexBIndex]).toBe(vertexA);
    expect(nextVertices[vertexAIndex][vertexGIndex]).toBe(vertexE);
    expect(nextVertices[vertexAIndex][vertexCIndex]).toBe(vertexA);
    expect(nextVertices[vertexAIndex][vertexAIndex]).toBe(null);
    expect(nextVertices[vertexAIndex][vertexHIndex]).toBe(null);
  });

  it('should find minimum paths to all vertices for directed graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB, 3);
    const edgeBA = new GraphEdge(vertexB, vertexA, 8);
    const edgeAD = new GraphEdge(vertexA, vertexD, 7);
    const edgeDA = new GraphEdge(vertexD, vertexA, 2);
    const edgeBC = new GraphEdge(vertexB, vertexC, 2);
    const edgeCA = new GraphEdge(vertexC, vertexA, 5);
    const edgeCD = new GraphEdge(vertexC, vertexD, 1);

    const graph = new Graph(true);

    // Add vertices first just to have them in desired order.
    graph
      .addVertex(vertexA)
      .addVertex(vertexB)
      .addVertex(vertexC)
      .addVertex(vertexD);

    // Now, when vertices are in correct order let's add edges.
    graph
      .addEdge(edgeAB)
      .addEdge(edgeBA)
      .addEdge(edgeAD)
      .addEdge(edgeDA)
      .addEdge(edgeBC)
      .addEdge(edgeCA)
      .addEdge(edgeCD);

    const { distances, nextVertices } = floydWarshall(graph);

    const vertices = graph.getAllVertices();

    const vertexAIndex = vertices.indexOf(vertexA);
    const vertexBIndex = vertices.indexOf(vertexB);
    const vertexCIndex = vertices.indexOf(vertexC);
    const vertexDIndex = vertices.indexOf(vertexD);

    expect(distances[vertexAIndex][vertexAIndex]).toBe(0);
    expect(distances[vertexAIndex][vertexBIndex]).toBe(3);
    expect(distances[vertexAIndex][vertexCIndex]).toBe(5);
    expect(distances[vertexAIndex][vertexDIndex]).toBe(6);

    expect(distances).toEqual([
      [0, 3, 5, 6],
      [5, 0, 2, 3],
      [3, 6, 0, 1],
      [2, 5, 7, 0],
    ]);

    expect(nextVertices[vertexAIndex][vertexDIndex]).toBe(vertexC);
    expect(nextVertices[vertexAIndex][vertexCIndex]).toBe(vertexB);
    expect(nextVertices[vertexBIndex][vertexDIndex]).toBe(vertexC);
    expect(nextVertices[vertexAIndex][vertexAIndex]).toBe(null);
    expect(nextVertices[vertexAIndex][vertexBIndex]).toBe(vertexA);
  });

  it('should find minimum paths to all vertices for directed graph with negative edge weights', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeFE = new GraphEdge(vertexF, vertexE, 8);
    const edgeFA = new GraphEdge(vertexF, vertexA, 10);
    const edgeED = new GraphEdge(vertexE, vertexD, 1);
    const edgeDA = new GraphEdge(vertexD, vertexA, -4);
    const edgeDC = new GraphEdge(vertexD, vertexC, -1);
    const edgeAC = new GraphEdge(vertexA, vertexC, 2);
    const edgeCB = new GraphEdge(vertexC, vertexB, -2);
    const edgeBA = new GraphEdge(vertexB, vertexA, 1);

    const graph = new Graph(true);

    // Add vertices first just to have them in desired order.
    graph
      .addVertex(vertexA)
      .addVertex(vertexB)
      .addVertex(vertexC)
      .addVertex(vertexD)
      .addVertex(vertexE)
      .addVertex(vertexF)
      .addVertex(vertexG);

    // Now, when vertices are in correct order let's add edges.
    graph
      .addEdge(edgeFE)
      .addEdge(edgeFA)
      .addEdge(edgeED)
      .addEdge(edgeDA)
      .addEdge(edgeDC)
      .addEdge(edgeAC)
      .addEdge(edgeCB)
      .addEdge(edgeBA);

    const { distances, nextVertices } = floydWarshall(graph);

    const vertices = graph.getAllVertices();

    const vertexAIndex = vertices.indexOf(vertexA);
    const vertexBIndex = vertices.indexOf(vertexB);
    const vertexCIndex = vertices.indexOf(vertexC);
    const vertexDIndex = vertices.indexOf(vertexD);
    const vertexEIndex = vertices.indexOf(vertexE);
    const vertexGIndex = vertices.indexOf(vertexG);
    const vertexFIndex = vertices.indexOf(vertexF);

    expect(distances[vertexFIndex][vertexGIndex]).toBe(Infinity);
    expect(distances[vertexFIndex][vertexFIndex]).toBe(0);
    expect(distances[vertexFIndex][vertexAIndex]).toBe(5);
    expect(distances[vertexFIndex][vertexBIndex]).toBe(5);
    expect(distances[vertexFIndex][vertexCIndex]).toBe(7);
    expect(distances[vertexFIndex][vertexDIndex]).toBe(9);
    expect(distances[vertexFIndex][vertexEIndex]).toBe(8);

    expect(nextVertices[vertexFIndex][vertexGIndex]).toBe(null);
    expect(nextVertices[vertexFIndex][vertexFIndex]).toBe(null);
    expect(nextVertices[vertexAIndex][vertexBIndex]).toBe(vertexC);
    expect(nextVertices[vertexAIndex][vertexCIndex]).toBe(vertexA);
    expect(nextVertices[vertexFIndex][vertexBIndex]).toBe(vertexE);
    expect(nextVertices[vertexEIndex][vertexBIndex]).toBe(vertexD);
    expect(nextVertices[vertexDIndex][vertexBIndex]).toBe(vertexC);
    expect(nextVertices[vertexCIndex][vertexBIndex]).toBe(vertexC);
  });
});

describe('graphBridges', () => {
  it('should find bridges in simple graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeCD);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(3);
    expect(bridges[0].getKey()).toBe(edgeCD.getKey());
    expect(bridges[1].getKey()).toBe(edgeBC.getKey());
    expect(bridges[2].getKey()).toBe(edgeAB.getKey());
  });

  it('should find bridges in simple graph with back edge', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeAC = new GraphEdge(vertexA, vertexC);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(1);
    expect(bridges[0].getKey()).toBe(edgeCD.getKey());
  });

  it('should find bridges in graph', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeGF = new GraphEdge(vertexG, vertexF);
    const edgeFH = new GraphEdge(vertexF, vertexH);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeAC)
      .addEdge(edgeCD)
      .addEdge(edgeDE)
      .addEdge(edgeEG)
      .addEdge(edgeEF)
      .addEdge(edgeGF)
      .addEdge(edgeFH);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(3);
    expect(bridges[0].getKey()).toBe(edgeFH.getKey());
    expect(bridges[1].getKey()).toBe(edgeDE.getKey());
    expect(bridges[2].getKey()).toBe(edgeCD.getKey());
  });

  it('should find bridges in graph starting with different root vertex', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');
    const vertexH = new GraphVertex('H');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeEF = new GraphEdge(vertexE, vertexF);
    const edgeGF = new GraphEdge(vertexG, vertexF);
    const edgeFH = new GraphEdge(vertexF, vertexH);

    const graph = new Graph();

    graph
      .addEdge(edgeDE)
      .addEdge(edgeAB)
      .addEdge(edgeBC)
      .addEdge(edgeAC)
      .addEdge(edgeCD)
      .addEdge(edgeEG)
      .addEdge(edgeEF)
      .addEdge(edgeGF)
      .addEdge(edgeFH);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(3);
    expect(bridges[0].getKey()).toBe(edgeFH.getKey());
    expect(bridges[1].getKey()).toBe(edgeDE.getKey());
    expect(bridges[2].getKey()).toBe(edgeCD.getKey());
  });

  it('should find bridges in yet another graph #1', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeDE = new GraphEdge(vertexD, vertexE);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD)
      .addEdge(edgeDE);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(2);
    expect(bridges[0].getKey()).toBe(edgeDE.getKey());
    expect(bridges[1].getKey()).toBe(edgeCD.getKey());
  });

  it('should find bridges in yet another graph #2', () => {
    const vertexA = new GraphVertex('A');
    const vertexB = new GraphVertex('B');
    const vertexC = new GraphVertex('C');
    const vertexD = new GraphVertex('D');
    const vertexE = new GraphVertex('E');
    const vertexF = new GraphVertex('F');
    const vertexG = new GraphVertex('G');

    const edgeAB = new GraphEdge(vertexA, vertexB);
    const edgeAC = new GraphEdge(vertexA, vertexC);
    const edgeBC = new GraphEdge(vertexB, vertexC);
    const edgeCD = new GraphEdge(vertexC, vertexD);
    const edgeCE = new GraphEdge(vertexC, vertexE);
    const edgeCF = new GraphEdge(vertexC, vertexF);
    const edgeEG = new GraphEdge(vertexE, vertexG);
    const edgeFG = new GraphEdge(vertexF, vertexG);

    const graph = new Graph();

    graph
      .addEdge(edgeAB)
      .addEdge(edgeAC)
      .addEdge(edgeBC)
      .addEdge(edgeCD)
      .addEdge(edgeCE)
      .addEdge(edgeCF)
      .addEdge(edgeEG)
      .addEdge(edgeFG);

    const bridges = Object.values(graphBridges(graph));

    expect(bridges.length).toBe(1);
    expect(bridges[0].getKey()).toBe(edgeCD.getKey());
  });
});