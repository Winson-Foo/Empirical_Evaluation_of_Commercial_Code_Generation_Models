// import { alphaNumericalSort } from '../../../After_Refactor_3/javascript_algorithms_2/AlphaNumericalSort'
// No below file
// import { encrypt, decrypt } from '../../../After_Refactor_3/javascript_algorithms_2/AffineCipher'
import alphaNumericPalindrome from '../../../After_Refactor_3/javascript_algorithms_2/AlphaNumericPalindrome'
import { binaryInsertionSort } from '../../../After_Refactor_3/javascript_algorithms_2/BinaryInsertionSort'
import { binarySearch } from '../../../After_Refactor_3/javascript_algorithms_2/BinarySearch'
import binaryToDecimal from '../../../After_Refactor_3/javascript_algorithms_2/BinaryToDecimal'
import { bucketSort } from '../../../After_Refactor_3/javascript_algorithms_2/BucketSort'
import { checkCamelCase } from '../../../After_Refactor_3/javascript_algorithms_2/CheckCamelCase'
import { checkExceeding } from '../../../After_Refactor_3/javascript_algorithms_2/CheckExceeding'
import { checkWordOccurrence } from '../../../After_Refactor_3/javascript_algorithms_2/CheckWordOccurrence'
import Cone from '../../../After_Refactor_3/javascript_algorithms_2/Cone'
// coprime does not have test
import { createPermutations } from '../../../After_Refactor_3/javascript_algorithms_2/CreatePermutations'
import { DateDayDifference } from '../../../After_Refactor_3/javascript_algorithms_2/DateDayDifference'
import { DateToDay } from '../../../After_Refactor_3/javascript_algorithms_2/DateToDay'
import { decimalToBinary } from '../../../After_Refactor_3/javascript_algorithms_2/DecimalToBinary'
// Density does not have test
import { diceCoefficient } from '../../../After_Refactor_3/javascript_algorithms_2/DiceCoefficient'
import { eulersTotientFunction } from '../../../After_Refactor_3/javascript_algorithms_2/EulersTotientFunction'
import { extendedEuclideanGCD } from '../../../After_Refactor_3/javascript_algorithms_2/ExtendedEuclideanGCD'
import { fibonacci } from '../../../After_Refactor_3/javascript_algorithms_2/FibonacciNumber'
import { findMaxRecursion } from '../../../After_Refactor_3/javascript_algorithms_2/FindMaxRecursion'
import { shuffle } from '../../../After_Refactor_3/javascript_algorithms_2/FisherYatesShuffle'
// import { breadthFirstSearch, depthFirstSearch } from '../../../After_Refactor_3/javascript_algorithms_2/FloodFill'
import { GetEuclidGCD } from '../../../After_Refactor_3/javascript_algorithms_2/GetEuclidGCD'
import { isPalindromeIntegerNumber } from '../../../After_Refactor_3/javascript_algorithms_2/isPalindromeIntegerNumber'
// KyeFinder does not have test
import { LinearSieve } from '../../../After_Refactor_3/javascript_algorithms_2/LinearSieve'
import litersToUSGallons from '../../../After_Refactor_3/javascript_algorithms_2/LitersToUSGallons'
import lower from '../../../After_Refactor_3/javascript_algorithms_2/Lower'
// MaxHeap does not have test
// MaxNonAdjacentSum does not have test
// No below file
// import { meanAbsoluteDeviation } from '../../../After_Refactor_3/javascript_algorithms_2/MeanAbsoluteDeviation.js'
import { meanSquaredError } from '../../../After_Refactor_3/javascript_algorithms_2/MeanSquareError'
import { radixSort } from '../../../After_Refactor_3/javascript_algorithms_2/RadixSort'
// ReverseNumber does not have test
import { approximatelyEqualHsv, hsvToRgb, rgbToHsv } from '../../../After_Refactor_3/javascript_algorithms_2/RgbHsvConversion'
import { isScramble } from '../../../After_Refactor_3/javascript_algorithms_2/ScrambleStrings'
import { SegmentTree } from '../../../After_Refactor_3/javascript_algorithms_2/SegmentTree'
import { shellSort } from '../../../After_Refactor_3/javascript_algorithms_2/ShellSort'
import { sieveOfEratosthenes } from '../../../After_Refactor_3/javascript_algorithms_2/SieveOfEratosthenes'
import Sphere from '../../../After_Refactor_3/javascript_algorithms_2/Sphere'
import { squareRootLogarithmic } from '../../../After_Refactor_3/javascript_algorithms_2/SquareRootLogarithmic'
import { stoogeSort } from '../../../After_Refactor_3/javascript_algorithms_2/StoogeSort'
// Topological sort does not have test
import { tribonacci } from '../../../After_Refactor_3/javascript_algorithms_2/TribonacciNumber'
// import upper from '../../../After_Refactor_3/javascript_algorithms_2/Upper'
import { validateCreditCard } from '../../../After_Refactor_3/javascript_algorithms_2/ValidateCreditCard'
import * as volume from '../../../After_Refactor_3/javascript_algorithms_2/Volume'
import XORCipher from '../../../After_Refactor_3/javascript_algorithms_2/XORCipher'

describe('alphaNumericalComparer', () => {
  test('given array of eng symbols return correct sorted array', () => {
    const src = ['b', 'a', 'c']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['a', 'b', 'c'])
  })

  test('given array of numbers return correct sorted array', () => {
    const src = ['15', '0', '5']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['0', '5', '15'])
  })

  test('correct sort with numbers and strings', () => {
    const src = ['3', 'a1b15c', 'z', 'a1b14c']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['3', 'a1b14c', 'a1b15c', 'z'])
  })

  test('correct sort with long numbers', () => {
    const src = ['abc999999999999999999999999999999999cba', 'abc999999999999999999999999999999990cba', 'ab']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['ab', 'abc999999999999999999999999999999990cba', 'abc999999999999999999999999999999999cba'])
  })

  test('correct sort with z prefix', () => {
    const src = ['z', 'abc003def', 'abc1def', 'a']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['a', 'abc1def', 'abc003def', 'z'])
  })

  test('correct sort with other language', () => {
    const src = ['а10б', 'а2б', 'в10г', 'в05г']
    src.sort(alphaNumericalSort)
    expect(src).toEqual(['а2б', 'а10б', 'в05г', 'в10г'])
  })
})

describe('Test Affine Cipher', () => {
  it('Test - 1, Pass invalid input to encrypt function', () => {
    expect(() => encrypt(null, null, null)).toThrow()
    expect(() => encrypt('null', null, null)).toThrow()
    expect(() => encrypt('null', 1, null)).toThrow()
    expect(() => encrypt('null', null, 1)).toThrow()
    expect(() => encrypt('null', 2, 1)).toThrow()
    expect(() => encrypt('null', 4, 1)).toThrow()
  })

  it('Test - 2, Pass invalid input to decrypt function', () => {
    expect(() => decrypt(null, null, null)).toThrow()
    expect(() => decrypt('null', null, null)).toThrow()
    expect(() => decrypt('null', 1, null)).toThrow()
    expect(() => decrypt('null', null, 1)).toThrow()
    expect(() => encrypt('null', 2, 1)).toThrow()
    expect(() => encrypt('null', 4, 1)).toThrow()
  })

  it('Test - 3 Pass string value to encrypt and decrypt function', () => {
    expect(decrypt(encrypt('HELLO WORLD', 5, 8), 5, 8)).toBe('HELLO WORLD')
    expect(decrypt(encrypt('ABC DEF', 3, 5), 3, 5)).toBe('ABC DEF')
    expect(decrypt(encrypt('Brown fox jump over the fence', 7, 3), 7, 3)).toBe(
      'BROWN FOX JUMP OVER THE FENCE'
    )
  })
})

describe('Testing the alpha numeric palindrome', () => {
  // should return true if the given string has alphanumeric characters that are palindrome irrespective of case and symbols
  it('Testing with valid alphabetic palindrome', () => {
    expect(alphaNumericPalindrome('eye')).toBe(true)
    expect(alphaNumericPalindrome('Madam')).toBe(true)
    expect(alphaNumericPalindrome('race CAR')).toBe(true)
    expect(alphaNumericPalindrome('A man, a plan, a canal. Panama')).toBe(true)
  })

  it('Testing with number and symbol', () => {
    expect(alphaNumericPalindrome('0_0 (: /-:) 0-0')).toBe(true)
    expect(alphaNumericPalindrome('03_|53411435|_30')).toBe(true)
  })

  it('Testing with alphabets and symbols', () => {
    expect(alphaNumericPalindrome('five|_/|evif')).toBe(true)
    expect(alphaNumericPalindrome('five|_/|four')).toBe(false)
  })
})

describe('BinaryInsertionSort', () => {
  it('should sort arrays correctly', () => {
    expect(binaryInsertionSort([5, 4, 3, 2, 1])).toEqual([1, 2, 3, 4, 5])
    expect(binaryInsertionSort([7, 9, 4, 3, 5])).toEqual([3, 4, 5, 7, 9])
  })
})

describe('BinarySearch', () => {
  const arr = [2, 3, 4, 10, 25, 40, 45, 60, 100, 501, 700, 755, 800, 999]
  const low = 0
  const high = arr.length - 1

  it('should return index 3 for searchValue 10', () => {
    const searchValue = 10
    expect(binarySearch(arr, searchValue, low, high)).toBe(3)
  })

  it('should return index 0 for searchValue 2', () => {
    const searchValue = 2
    expect(binarySearch(arr, searchValue, low, high)).toBe(0)
  })

  it('should return index 13 for searchValue 999', () => {
    const searchValue = 999
    expect(binarySearch(arr, searchValue, low, high)).toBe(13)
  })

  it('should return -1 for searchValue 1', () => {
    const searchValue = 1
    expect(binarySearch(arr, searchValue, low, high)).toBe(-1)
  })

  it('should return -1 for searchValue 1000', () => {
    const searchValue = 1000
    expect(binarySearch(arr, searchValue, low, high)).toBe(-1)
  })
})

describe('BinaryToDecimal', () => {
  it('expects to return correct decimal value', () => {
    expect(binaryToDecimal('1000')).toBe(8)
  })

  it('expects to return correct hexadecimal value for more than one hex digit', () => {
    expect(binaryToDecimal('01101000')).toBe(104)
  })

  it('expects to return correct hexadecimal value for padding-required binary', () => {
    expect(binaryToDecimal('1000101')).toBe(69)
  })
})

describe('Tests for bucketSort function', () => {
  it('should correctly sort an input list that is sorted backwards', () => {
    const array = [5, 4, 3, 2, 1]
    expect(bucketSort(array)).toEqual([1, 2, 3, 4, 5])
  })

  it('should correctly sort an input list that is unsorted', () => {
    const array = [15, 24, 3, 2224, 1]
    expect(bucketSort(array)).toEqual([1, 3, 15, 24, 2224])
  })

  describe('Variations of input array lengths', () => {
    it('should return an empty list with the input list is an empty list', () => {
      expect(bucketSort([])).toEqual([])
    })

    it('should correctly sort an input list of length 1', () => {
      expect(bucketSort([100])).toEqual([100])
    })

    it('should correctly sort an input list of an odd length', () => {
      expect(bucketSort([101, -10, 321])).toEqual([-10, 101, 321])
    })

    it('should correctly sort an input list of an even length', () => {
      expect(bucketSort([40, 42, 56, 45, 12, 3])).toEqual([3, 12, 40, 42, 45, 56])
    })
  })

  describe('Variations of input array elements', () => {
    it('should correctly sort an input list that contains only positive numbers', () => {
      expect(bucketSort([50, 33, 11, 2])).toEqual([2, 11, 33, 50])
    })

    it('should correctly sort an input list that contains only negative numbers', () => {
      expect(bucketSort([-1, -21, -2, -35])).toEqual([-35, -21, -2, -1])
    })

    it('should correctly sort an input list that contains only a mix of positive and negative numbers', () => {
      expect(bucketSort([-40, 42, 56, -45, 12, -3])).toEqual([-45, -40, -3, 12, 42, 56])
    })

    it('should correctly sort an input list that contains only whole numbers', () => {
      expect(bucketSort([11, 3, 12, 4, -15])).toEqual([-15, 3, 4, 11, 12])
    })

    it('should correctly sort an input list that contains only decimal numbers', () => {
      expect(bucketSort([1.0, 1.42, 2.56, 33.45, 13.12, 2.3])).toEqual([1.0, 1.42, 2.3, 2.56, 13.12, 33.45])
    })

    it('should correctly sort an input list that contains only a mix of whole and decimal', () => {
      expect(bucketSort([32.40, 12.42, 56, 45, 12, 3])).toEqual([3, 12, 12.42, 32.40, 45, 56])
    })

    it('should correctly sort an input list that contains only fractional numbers', () => {
      expect(bucketSort([0.98, 0.4259, 0.56, -0.456, -0.12, 0.322])).toEqual([-0.456, -0.12, 0.322, 0.4259, 0.56, 0.98])
    })

    it('should correctly sort an input list that contains only a mix of whole, decimal, and fractional', () => {
      expect(bucketSort([-40, -0.222, 5.6, -4.5, 12, 0.333])).toEqual([-40, -4.5, -0.222, 0.333, 5.6, 12])
    })

    it('should correctly sort an input list that contains duplicates', () => {
      expect(bucketSort([4, 3, 4, 2, 1, 2])).toEqual([1, 2, 2, 3, 4, 4])
    })
  })
})

describe('checkCamelCase', () => {
  it('expect to throw an error if input is not a string', () => {
    expect(() => checkCamelCase(null)).toThrow()
  })

  it('expects to return true if the input is in camel case format', () => {
    const value = 'dockerBuild'
    const result = checkCamelCase(value)
    expect(result).toBe(true)
  })

  it('expects to return false if the input is not in camel case format', () => {
    const value = 'docker_build'
    const result = checkCamelCase(value)
    expect(result).toBe(false)
  })
})

describe('Testing CheckExceeding function', () => {
  it('Testing the invalid types', () => {
    expect(() => checkExceeding(Math.random())).toThrow('Argument is not a string')
    expect(() => checkExceeding(null)).toThrow('Argument is not a string')
    expect(() => checkExceeding(false)).toThrow('Argument is not a string')
    expect(() => checkExceeding(false)).toThrow('Argument is not a string')
  })

  it('Testing with empty string', () => {
    expect(checkExceeding('')).toBe(true)
  })

  it('Testing with linear alphabets', () => {
    expect(checkExceeding('a b c d e ')).toBe(true)
    expect(checkExceeding('f g h i j ')).toBe(true)
    expect(checkExceeding('k l m n o ')).toBe(true)
    expect(checkExceeding('p q r s t ')).toBe(true)
    expect(checkExceeding('u v w x y z')).toBe(true)
  })

  it('Testing not exceeding words', () => {
    expect(checkExceeding('Hello')).toBe(false)
    expect(checkExceeding('world')).toBe(false)
    expect(checkExceeding('update')).toBe(false)
    expect(checkExceeding('university')).toBe(false)
    expect(checkExceeding('dog')).toBe(false)
    expect(checkExceeding('exceeding')).toBe(false)
    expect(checkExceeding('resolved')).toBe(false)
    expect(checkExceeding('future')).toBe(false)
    expect(checkExceeding('fixed')).toBe(false)
    expect(checkExceeding('codes')).toBe(false)
    expect(checkExceeding('facebook')).toBe(false)
    expect(checkExceeding('vscode')).toBe(false)
  })

  it('Testing exceeding words', () => {
    expect(checkExceeding('bee')).toBe(true) // [ 3 ]
    expect(checkExceeding('can')).toBe(true) // [ 2, 13 ]
    expect(checkExceeding('good')).toBe(true) //  [ 8, 11 ]
    expect(checkExceeding('bad')).toBe(true) // [ 1, 3 ]
    expect(checkExceeding('play')).toBe(true) // [ 4, 11, 24 ]
    expect(checkExceeding('delete')).toBe(true) // [1, 7, 7, 15, 15]
  })
})

describe('Testing checkWordOccurrence', () => {
  it('expects throw on insert wrong string', () => {
    const value = 123

    expect(() => checkWordOccurrence(value)).toThrow()
  })

  it('expect throw on insert wrong param for case sensitive', () => {
    const value = 'hello'

    expect(() => checkWordOccurrence(value, value)).toThrow()
  })

  it('check occurrence with case sensitive', () => {
    const stringToTest = 'The quick brown fox jumps over the lazy dog'
    const expectResult = { The: 1, quick: 1, brown: 1, fox: 1, jumps: 1, over: 1, the: 1, lazy: 1, dog: 1 }

    expect(checkWordOccurrence(stringToTest)).toEqual(expectResult)
  })

  it('check occurrence with case insensitive', () => {
    const stringToTest = 'The quick brown fox jumps over the lazy dog'
    const expectResult = { the: 2, quick: 1, brown: 1, fox: 1, jumps: 1, over: 1, lazy: 1, dog: 1 }

    expect(checkWordOccurrence(stringToTest, true)).toEqual(expectResult)
  })
})

const cone = new Cone(3, 5)

test('The Volume of a cone with base radius equal to 3 and height equal to 5', () => {
  expect(parseFloat(cone.volume().toFixed(2))).toEqual(47.12)
})

test('The Surface Area of a cone with base radius equal to 3 and height equal to 5', () => {
  expect(parseFloat(cone.surfaceArea().toFixed(2))).toEqual(83.23)
})

describe('createPermutations', () => {
  it('expects to generate 6 different combinations', () => {
    const text = 'abc'
    const SUT = createPermutations(text)
    expect(SUT).toStrictEqual(['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])
  })
  it('expects to generate 2 different combinations', () => {
    const text = '12'
    const SUT = createPermutations(text)
    expect(SUT).toStrictEqual(['12', '21'])
  })
})

test('The difference between 17/08/2002 & 10/10/2020 is 6630', () => {
  const res = DateDayDifference('17/08/2002', '10/10/2020')
  expect(res).toBe(6630)
})

test('The difference between 18/02/2001 & 16/03/2022 is 7696', () => {
  const res = DateDayDifference('18/02/2001', '16/03/2022')
  expect(res).toBe(7696)
})

test('The difference between 11/11/2011 & 12/12/2012 is 398', () => {
  const res = DateDayDifference('11/11/2011', '12/12/2012')
  expect(res).toBe(398)
})

test('The difference between 01/01/2001 & 16/03/2011 is 3727', () => {
  const res = DateDayDifference('01/01/2001', '16/03/2011')
  expect(res).toBe(3727)
})

test('The date 18/02/2001 is Sunday', () => {
  const res = DateToDay('18/02/2001')
  expect(res).toBe('Sunday')
})

test('The date 18/12/2020 is Friday', () => {
  const res = DateToDay('18/12/2020')
  expect(res).toBe('Friday')
})

test('The date 12/12/2012 is Wednesday', () => {
  const res = DateToDay('12/12/2012')
  expect(res).toBe('Wednesday')
})
test('The date 01/01/2001 is Monday', () => {
  const res = DateToDay('01/01/2001')
  expect(res).toBe('Monday')
})

test('The date 1/1/2020 is Wednesday', () => {
  const res = DateToDay('1/1/2020')
  expect(res).toBe('Wednesday')
})

test('The date 2/3/2014 is Sunday', () => {
  const res = DateToDay('2/3/2014')
  expect(res).toBe('Sunday')
})

test('The date 28/2/2017 is Tuesday', () => {
  const res = DateToDay('28/2/2017')
  expect(res).toBe('Tuesday')
})


test('The Binary representation of 35 is 100011', () => {
  const res = decimalToBinary(35)
  expect(res).toBe('100011')
})

test('The Binary representation of 1 is 1', () => {
  const res = decimalToBinary(1)
  expect(res).toBe('1')
})

test('The Binary representation of 1000 is 1111101000', () => {
  const res = decimalToBinary(1000)
  expect(res).toBe('1111101000')
})

test('The Binary representation of 2 is 10', () => {
  const res = decimalToBinary(2)
  expect(res).toBe('10')
})

test('The Binary representation of 17 is 10001', () => {
  const res = decimalToBinary(17)
  expect(res).toBe('10001')
})

describe('diceCoefficient', () => {
  it('should calculate edit distance between two strings', () => {
    // equal strings return 1 (max possible value)
    expect(diceCoefficient('abc', 'abc')).toBe(1)
    expect(diceCoefficient('', '')).toBe(1)

    // string length needs to be at least 2 (unless equal)
    expect(diceCoefficient('a', '')).toBe(0)
    expect(diceCoefficient('', 'a')).toBe(0)

    expect(diceCoefficient('skate', 'ate')).toBe(0.66)

    expect(diceCoefficient('money', 'honey')).toBe(0.75)

    expect(diceCoefficient('love', 'hate')).toBe(0)

    expect(diceCoefficient('skilled', 'killed')).toBe(0.9)
  })
})

describe('eulersTotientFunction', () => {
  it('is a function', () => {
    expect(typeof eulersTotientFunction).toEqual('function')
  })
  it('should return the phi of a given number', () => {
    const phiOfNumber = eulersTotientFunction(10)
    expect(phiOfNumber).toBe(4)
  })
})

describe('extendedEuclideanGCD', () => {
  it('should return valid values in order for positive arguments', () => {
    expect(extendedEuclideanGCD(240, 46)).toMatchObject([2, -9, 47])
    expect(extendedEuclideanGCD(46, 240)).toMatchObject([2, 47, -9])
  })
  it('should give error on non-positive arguments', () => {
    expect(() => extendedEuclideanGCD(0, 240)).toThrowError(new TypeError('Must be positive numbers'))
    expect(() => extendedEuclideanGCD(46, -240)).toThrowError(new TypeError('Must be positive numbers'))
  })
  it('should give error on non-numeric arguments', () => {
    expect(() => extendedEuclideanGCD('240', 46)).toThrowError(new TypeError('Not a Number'))
    expect(() => extendedEuclideanGCD([240, 46])).toThrowError(new TypeError('Not a Number'))
  })
})

describe('Testing FibonacciNumber', () => {
  it('Testing for invalid type', () => {
    expect(() => fibonacci('0')).toThrowError()
    expect(() => fibonacci('12')).toThrowError()
    expect(() => fibonacci(true)).toThrowError()
  })

  it('fibonacci of 0', () => {
    expect(fibonacci(0)).toBe(0)
  })

  it('fibonacci of 1', () => {
    expect(fibonacci(1)).toBe(1)
  })

  it('fibonacci of 10', () => {
    expect(fibonacci(10)).toBe(55)
  })

  it('fibonacci of 25', () => {
    expect(fibonacci(25)).toBe(75025)
  })
})

describe('Test findMaxRecursion function', () => {
  const positiveAndNegativeArray = [1, 2, 4, 5, -1, -2, -4, -5]
  const positiveAndNegativeArray1 = [10, 40, 100, 20, -10, -40, -100, -20]

  const positiveArray = [1, 2, 4, 5]
  const positiveArray1 = [10, 40, 100, 20]

  const negativeArray = [-1, -2, -4, -5]
  const negativeArray1 = [-10, -40, -100, -20]

  const zeroArray = [0, 0, 0, 0]
  const emptyArray = []

  it('Testing with positive arrays', () => {
    expect(findMaxRecursion(positiveArray, 0, positiveArray.length - 1)).toBe(5)
    expect(findMaxRecursion(positiveArray1, 0, positiveArray1.length - 1)).toBe(
      100
    )
  })

  it('Testing with negative arrays', () => {
    expect(findMaxRecursion(negativeArray, 0, negativeArray.length - 1)).toBe(
      -1
    )
    expect(findMaxRecursion(negativeArray1, 0, negativeArray1.length - 1)).toBe(
      -10
    )
  })

  it('Testing with positive and negative arrays', () => {
    expect(
      findMaxRecursion(
        positiveAndNegativeArray,
        0,
        positiveAndNegativeArray.length - 1
      )
    ).toBe(5)
    expect(
      findMaxRecursion(
        positiveAndNegativeArray1,
        0,
        positiveAndNegativeArray1.length - 1
      )
    ).toBe(100)
  })

  it('Testing with zero arrays', () => {
    expect(findMaxRecursion(zeroArray, 0, zeroArray.length - 1)).toBe(0)
  })

  it('Testing with empty arrays', () => {
    expect(findMaxRecursion(emptyArray, 0, emptyArray.length - 1)).toBe(
      undefined
    )
  })
})

describe('shuffle', () => {
  it('expects to have a new array with same size', () => {
    const fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    const mixedArray = shuffle(fibonacci)

    expect(mixedArray).toHaveLength(fibonacci.length)
  })

  it('expects to have a new array with same values', () => {
    const fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    const mixedArray = shuffle(fibonacci)

    expect(mixedArray).toContain(0)
    expect(mixedArray).toContain(1)
    expect(mixedArray).toContain(2)
    expect(mixedArray).toContain(3)
    expect(mixedArray).toContain(5)
    expect(mixedArray).toContain(8)
    expect(mixedArray).toContain(13)
    expect(mixedArray).toContain(21)
    expect(mixedArray).toContain(34)
    expect(mixedArray).toContain(55)
    expect(mixedArray).toContain(89)
  })
})

// // some constants
// const black = [0, 0, 0]
// const green = [0, 255, 0]
// const violet = [255, 0, 255]
// const white = [255, 255, 255]
// const orange = [255, 128, 0]

// describe('FloodFill', () => {
//   it('should calculate the correct colors using breadth-first approach', () => {
//     expect(testBreadthFirst([1, 1], green, orange, [1, 1])).toEqual(orange)
//     expect(testBreadthFirst([1, 1], green, orange, [0, 1])).toEqual(violet)
//     expect(testBreadthFirst([1, 1], green, orange, [6, 4])).toEqual(white)
//   })

//   it('should calculate the correct colors using depth-first approach', () => {
//     expect(testDepthFirst([1, 1], green, orange, [1, 1])).toEqual(orange)
//     expect(testDepthFirst([1, 1], green, orange, [0, 1])).toEqual(violet)
//     expect(testDepthFirst([1, 1], green, orange, [6, 4])).toEqual(white)
//   })
// })

// /**
//  * Utility-function to test the function "breadthFirstSearch".
//  *
//  * @param fillLocation The start location on the image where the flood fill is applied.
//  * @param targetColor The old color to be replaced.
//  * @param replacementColor The new color to replace the old one.
//  * @param testLocation The location of the color to be checked.
//  * @return The color at testLocation.
//  */
// function testBreadthFirst (fillLocation, targetColor, replacementColor, testLocation) {
//   const rgbData = generateTestRgbData()
//   breadthFirstSearch(rgbData, fillLocation, targetColor, replacementColor)
//   return rgbData[testLocation[0]][testLocation[1]]
// }

/**
 * Utility-function to test the function "depthFirstSearch".
 *
 * @param fillLocation The start location on the image where the flood fill is applied.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 * @param testLocation The location of the color to be checked.
 * @return The color at testLocation.
 */
function testDepthFirst (fillLocation, targetColor, replacementColor, testLocation) {// eslint-disable-line
  const rgbData = generateTestRgbData()
  depthFirstSearch(rgbData, fillLocation, targetColor, replacementColor)
  return rgbData[testLocation[0]][testLocation[1]]
}

/**
 * Generates the rgbData-matrix for the tests.
 *
 * @return example rgbData-matrix.
 */
function generateTestRgbData () {
  const layout = [
    [violet, violet, green, green, black, green, green],
    [violet, green, green, black, green, green, green],
    [green, green, green, black, green, green, green],
    [black, black, green, black, white, white, green],
    [violet, violet, black, violet, violet, white, white],
    [green, green, green, violet, violet, violet, violet],
    [violet, violet, violet, violet, violet, violet, violet]
  ]

  // transpose layout-matrix so the x-index comes before the y-index
  const transposed = []
  for (let x = 0; x < layout[0].length; x++) {
    transposed[x] = []
    for (let y = 0; y < layout.length; y++) {
      transposed[x][y] = layout[y][x]
    }
  }

  return transposed
}

function testEuclidGCD (n, m, expected) {
  test('Testing on ' + n + ' and ' + m + '!', () => {
    expect(GetEuclidGCD(n, m)).toBe(expected)
  })
}

testEuclidGCD(5, 20, 5)
testEuclidGCD(109, 902, 1)
testEuclidGCD(290, 780, 10)
testEuclidGCD(104, 156, 52)


describe('isPalindromeIntegerNumber', () => {
  it('expects to return true when length of input is 1', () => {
    expect(isPalindromeIntegerNumber(6)).toEqual(true)
  })

  it('expects to return true when input is palindrome', () => {
    expect(isPalindromeIntegerNumber(121)).toEqual(true)
    expect(isPalindromeIntegerNumber(12321)).toEqual(true)
    expect(isPalindromeIntegerNumber(1221)).toEqual(true)
  })

  it('expects to return false when input is not palindrome', () => {
    expect(isPalindromeIntegerNumber(189)).toEqual(false)
  })

  it('expects to return false when input is minus', () => {
    expect(isPalindromeIntegerNumber(-121)).toEqual(false)
    expect(isPalindromeIntegerNumber(-12321)).toEqual(false)
  })

  it('expects to return false when input is not integer number', () => {
    expect(isPalindromeIntegerNumber(123.456)).toEqual(false)
  })

  it('expects to throw error when input is not a number', () => {
    expect(() => isPalindromeIntegerNumber(undefined)).toThrowError()
    expect(() => isPalindromeIntegerNumber({ key: 'val' })).toThrowError()
    expect(() => isPalindromeIntegerNumber([])).toThrowError()
  })
})

describe('LinearSieve', () => {
  it('should return primes below 100', () => {
    expect(LinearSieve(100)).toEqual([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
  })
})

test('Convert 50 liters to US gallons', () => {
  expect(parseFloat(litersToUSGallons(50).toFixed(2))).toBe(13.21)
})


// describe('Testing the Lower function', () => {
//   it('Test 1: Check by invalid type', () => {
//     expect(() => lower(345)).toThrowError()
//     expect(() => lower(true)).toThrowError()
//     expect(() => lower(null)).toThrowError()
//   })

//   it('Test 2: Check by uppercase string', () => {
//     expect(lower('WORLD')).toBe('world')
//     expect(lower('Hello_WORLD')).toBe('hello_world')
//   })

//   it('Test 3: Check by lowercase string', () => {
//     expect(lower('hello')).toBe('hello')
//     expect(lower('hello_world')).toBe('hello_world')
//   })
// })

describe('tests for mean absolute deviation', () => {
  it('should be a function', () => {
    expect(typeof meanAbsoluteDeviation).toEqual('function')
  })

  it('should throw an invalid input error', () => {
    expect(() => meanAbsoluteDeviation('fgh')).toThrow()
  })

  it('should return the mean absolute deviation of an array of numbers', () => {
    const meanAbDev = meanAbsoluteDeviation([2, 34, 5, 0, -2])
    expect(meanAbDev).toBe(10.479999999999999)
  })
})

describe('meanSquareError', () => {
  it('should throw an error on non-array arguments', () => {
    expect(() => meanSquaredError(1, 4)).toThrow('Argument must be an Array')
  })

  it('should throw an error on non equal length ', () => {
    const firstArr = [1, 2, 3, 4, 5]
    const secondArr = [1, 2, 3]
    expect(() => meanSquaredError(firstArr, secondArr)).toThrow(
      'The two lists must be of equal length'
    )
  })

  it('should return the mean square error of two equal length arrays', () => {
    const firstArr = [1, 2, 3, 4, 5]
    const secondArr = [1, 3, 5, 6, 7]
    expect(meanSquaredError(firstArr, secondArr)).toBe(2.6)
  })
})

test('The RadixSort of the array [4, 3, 2, 1] is [1, 2, 3, 4]', () => {
  const arr = [4, 3, 2, 1]
  const res = radixSort(arr, 10)
  expect(res).toEqual([1, 2, 3, 4])
})

test('The RadixSort of the array [] is []', () => {
  const arr = []
  const res = radixSort(arr, 10)
  expect(res).toEqual([])
})

test('The RadixSort of the array [14, 16, 10, 12] is [10, 12, 14, 16]', () => {
  const arr = [14, 16, 10, 12]
  const res = radixSort(arr, 10)
  expect(res).toEqual([10, 12, 14, 16])
})


describe('hsvToRgb', () => {
  // Expected RGB-values taken from https://www.rapidtables.com/convert/color/hsv-to-rgb.html
  it('should calculate the correct RGB values', () => {
    expect(hsvToRgb(0, 0, 0)).toEqual([0, 0, 0])
    expect(hsvToRgb(0, 0, 1)).toEqual([255, 255, 255])
    expect(hsvToRgb(0, 1, 1)).toEqual([255, 0, 0])
    expect(hsvToRgb(60, 1, 1)).toEqual([255, 255, 0])
    expect(hsvToRgb(120, 1, 1)).toEqual([0, 255, 0])
    expect(hsvToRgb(240, 1, 1)).toEqual([0, 0, 255])
    expect(hsvToRgb(300, 1, 1)).toEqual([255, 0, 255])
    expect(hsvToRgb(180, 0.5, 0.5)).toEqual([64, 128, 128])
    expect(hsvToRgb(234, 0.14, 0.88)).toEqual([193, 196, 224])
    expect(hsvToRgb(330, 0.75, 0.5)).toEqual([128, 32, 80])
  })
})

describe('rgbToHsv', () => {
  // "approximatelyEqualHsv" needed because of small deviations due to rounding for the RGB-values
  it('should calculate the correct HSV values', () => {
    expect(approximatelyEqualHsv(rgbToHsv(0, 0, 0), [0, 0, 0])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(255, 255, 255), [0, 0, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(255, 0, 0), [0, 1, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(255, 255, 0), [60, 1, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(0, 255, 0), [120, 1, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(0, 0, 255), [240, 1, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(255, 0, 255), [300, 1, 1])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(64, 128, 128), [180, 0.5, 0.5])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(193, 196, 224), [234, 0.14, 0.88])).toEqual(true)
    expect(approximatelyEqualHsv(rgbToHsv(128, 32, 80), [330, 0.75, 0.5])).toEqual(true)
  })
})

describe('ScrambleStrings', () => {
  it('expects to return true for same string', () => {
    expect(isScramble('a', 'a')).toBe(true)
  })

  it('expects to return false for non-scrambled strings', () => {
    expect(isScramble('abcde', 'caebd')).toBe(false)
  })

  it('expects to return true for scrambled strings', () => {
    expect(isScramble('great', 'rgeat')).toBe(true)
  })
})

describe('SegmentTree sum test', () => {
  const a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

  const segment = new SegmentTree(a)

  it('init sum check', () => {
    expect(segment.query(0, 2)).toBe(6)
  })

  it('init sum check', () => {
    segment.update(2, 1)
    expect(segment.query(0, 2)).toBe(4)
  })
})

test('The ShellSort of the array [5, 4, 3, 2, 1] is [1, 2, 3, 4, 5]', () => {
  const arr = [5, 4, 3, 2, 1]
  const res = shellSort(arr)
  expect(res).toEqual([1, 2, 3, 4, 5])
})

test('The ShellSort of the array [] is []', () => {
  const arr = []
  const res = shellSort(arr)
  expect(res).toEqual([])
})

test('The ShellSort of the array [15, 24, 31, 42, 11] is [11, 15, 24, 31, 42]', () => {
  const arr = [15, 24, 31, 42, 11]
  const res = shellSort(arr)
  expect(res).toEqual([11, 15, 24, 31, 42])
})

test('The ShellSort of the array [121, 190, 169] is [121, 169, 190]', () => {
  const arr = [121, 190, 169]
  const res = shellSort(arr)
  expect(res).toEqual([121, 169, 190])
})

describe('SieveOfEratosthenes', () => {
  it('Primes till 0', () => {
    expect(sieveOfEratosthenes(0)).toEqual([])
  })

  it('Primes till 1', () => {
    expect(sieveOfEratosthenes(1)).toEqual([])
  })

  it('Primes till 10', () => {
    expect(sieveOfEratosthenes(10)).toEqual([2, 3, 5, 7])
  })

  it('Primes till 23', () => {
    expect(sieveOfEratosthenes(23)).toEqual([2, 3, 5, 7, 11, 13, 17, 19, 23])
  })

  it('Primes till 70', () => {
    expect(sieveOfEratosthenes(70)).toEqual([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
  })
})

const sphere = new Sphere(3)

test('The Volume of a sphere with base radius equal to 3 and height equal to 5', () => {
  expect(parseFloat(sphere.volume().toFixed(2))).toEqual(113.1)
})

test('The Surface Area of a sphere with base radius equal to 3 and height equal to 5', () => {
  expect(parseFloat(sphere.surfaceArea().toFixed(2))).toEqual(113.1)
})

describe('SquareRootLogarithmic', () => {
  test('Finding the square root of a positive integer', () => {
    expect(squareRootLogarithmic(4)).toEqual(2)
    expect(squareRootLogarithmic(16)).toEqual(4)
    expect(squareRootLogarithmic(8)).toEqual(2)
  })
  test('Throwing an exception', () => {
    expect(() => squareRootLogarithmic('not a number')).toThrow()
    expect(() => squareRootLogarithmic(true)).toThrow()
  })
})

test('The StoogeSort of the array [1, 6, 4, 7, 2] is [1, 2, 4, 6, 7]', () => {
  const arr = [1, 6, 4, 7, 2]
  const res = stoogeSort(arr, 0, arr.length)
  expect(res).toEqual([1, 2, 4, 6, 7])
})

test('The StoogeSort of the array [] is []', () => {
  const arr = []
  const res = stoogeSort(arr, 0, arr.length)
  expect(res).toEqual([])
})

test('The StoogeSort of the array [46, 15, 49, 65, 23] is [15, 23, 46, 49, 65]', () => {
  const arr = [46, 15, 49, 65, 23]
  const res = stoogeSort(arr, 0, arr.length)
  expect(res).toEqual([15, 23, 46, 49, 65])
})

test('The StoogeSort of the array [136, 459, 132, 566, 465] is [132, 136, 459, 465, 566]', () => {
  const arr = [136, 459, 132, 566, 465]
  const res = stoogeSort(arr, 0, arr.length)
  expect(res).toEqual([132, 136, 459, 465, 566])
})

test('The StoogeSort of the array [45, 3, 156, 1, 56] is [1, 3, 45, 56, 156]', () => {
  const arr = [45, 3, 156, 1, 56]
  const res = stoogeSort(arr, 0, arr.length)
  expect(res).toEqual([1, 3, 45, 56, 156])
})


describe('TribonacciNumber', () => {
  it('tribonacci of 0', () => {
    expect(tribonacci(0)).toBe(0)
  })

  it('tribonacci of 1', () => {
    expect(tribonacci(1)).toBe(1)
  })

  it('tribonacci of 2', () => {
    expect(tribonacci(2)).toBe(1)
  })

  it('tribonacci of 10', () => {
    expect(tribonacci(10)).toBe(149)
  })

  it('tribonacci of 25', () => {
    expect(tribonacci(25)).toBe(1389537)
  })
})

describe('Testing the Upper function', () => {
  it('return uppercase strings', () => {
    expect(upper('hello')).toBe('HELLO')
    expect(upper('WORLD')).toBe('WORLD')
    expect(upper('hello_WORLD')).toBe('HELLO_WORLD')
  })
})

describe('Validate credit card number', () => {
  it('should throw error if card number is boolean', () => {
    const invalidCC = true
    expect(() => validateCreditCard(invalidCC)).toThrow(
      'The given value is not a string'
    )
  })
  it('returns true if the credit card number is valid', () => {
    const validCreditCard = '4111111111111111'
    const validationResult = validateCreditCard(validCreditCard)
    expect(validationResult).toBe(true)
  })
  it('should throw an error on non-numeric character in given credit card number', () => {
    const nonNumericCCNumbers = ['123ABCDEF', 'ABCDKDKD', 'ADS232']
    nonNumericCCNumbers.forEach(nonNumericCC => expect(() => validateCreditCard(nonNumericCC)).toThrow(
      `${nonNumericCC} is an invalid credit card number because ` + 'it has nonnumerical characters.'
    ))
  })
  it('should throw an error on credit card with invalid length', () => {
    const ccWithInvalidLength = ['41111', '4111111111111111111111']
    ccWithInvalidLength.forEach(invalidCC => expect(() => validateCreditCard(invalidCC)).toThrow(
      `${invalidCC} is an invalid credit card number because ` + 'of its length.'
    ))
  })
  it('should throw an error on credit card with invalid start substring', () => {
    const ccWithInvalidStartSubstring = ['12345678912345', '23456789123456', '789123456789123', '891234567891234', '912345678912345', '31345678912345', '32345678912345', '33345678912345', '38345678912345']
    ccWithInvalidStartSubstring.forEach(invalidCC => expect(() => validateCreditCard(invalidCC)).toThrow(
      `${invalidCC} is an invalid credit card number because ` + 'of its first two digits.'
    ))
  })
  it('should throw an error on credit card with luhn check fail', () => {
    const invalidCCs = ['411111111111111', '371211111111111', '49999999999999']
    invalidCCs.forEach(invalidCC => expect(() => validateCreditCard(invalidCC)).toThrow(
      `${invalidCC} is an invalid credit card number because ` + 'it fails the Luhn check.'
    ))
  })
})

test('Testing on volCuboid', () => {
  const volCuboid = volume.volCuboid(2.0, 5.0, 3)
  expect(volCuboid).toBe(30.0)
})

test('Testing on volCube', () => {
  const volCube = volume.volCube(2.0)
  expect(volCube).toBe(8.0)
})

test('Testing on volCone', () => {
  const volCone = volume.volCone(3.0, 8.0)
  expect(volCone).toBe(75.39822368615503)
})

test('Testing on volPyramid', () => {
  const volPyramid = volume.volPyramid(2.0, 3.0, 8.0)
  expect(volPyramid).toBe(16.0)
})

test('Testing on volCylinder', () => {
  const volCylinder = volume.volCylinder(3.0, 8.0)
  expect(volCylinder).toBe(226.1946710584651)
})

test('Testing on volTriangularPrism', () => {
  const volTriangularPrism = volume.volTriangularPrism(3.0, 6.0, 8.0)
  expect(volTriangularPrism).toBe(72.0)
})

test('Testing on volPentagonalPrism', () => {
  const volPentagonalPrism = volume.volPentagonalPrism(1.0, 4.0, 8.0)
  expect(volPentagonalPrism).toBe(80.0)
})

test('Testing on volSphere', () => {
  const volSphere = volume.volSphere(4.0)
  expect(volSphere).toBe(268.082573106329)
})

test('Testing on volHemisphere', () => {
  const volHemisphere = volume.volHemisphere(4.0)
  expect(volHemisphere).toBe(134.0412865531645)
})

describe('Testing XORCipher function', () => {
  it('Test - 1, passing a non-string as an argument', () => {
    expect(() => XORCipher(false, 0x345)).toThrow()
    expect(() => XORCipher(true, 123)).toThrow()
    expect(() => XORCipher(1n, 123n)).toThrow()
    expect(() => XORCipher(false, 0.34)).toThrow()
    expect(() => XORCipher({})).toThrow()
    expect(() => XORCipher([])).toThrow()
  })

  it('Test - 2, passing a string & number as an argument', () => {
    // NB: Node REPL might not output the null char '\x00' (charcode 0)
    expect(XORCipher('test string', 32)).toBe('TEST\x00STRING')
    expect(XORCipher('TEST\x00STRING', 32)).toBe('test string')
  })
})