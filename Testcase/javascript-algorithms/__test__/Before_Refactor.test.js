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
