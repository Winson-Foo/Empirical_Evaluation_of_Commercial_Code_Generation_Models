package com.google.android.exoplayer2.util;

import java.util.Arrays;

/**
 * An append-only, auto-growing {@code long[]}.
 */
public final class LongArray {

  private static final int DEFAULT_INITIAL_CAPACITY = 32;
  private static final String ERROR_INVALID_INDEX = "Invalid index %s, size is %s";

  private int size;
  private long[] values;

  /**
   * Constructs a new LongArray with the default initial capacity.
   */
  public LongArray() {
    this(DEFAULT_INITIAL_CAPACITY);
  }

  /**
   * Constructs a new LongArray with the specified initial capacity.
   *
   * @param initialCapacity the initial capacity of the array.
   */
  public LongArray(final int initialCapacity) {
    values = new long[initialCapacity];
  }

  /**
   * Appends a value to the end of the LongArray.
   *
   * @param value the value to append.
   */
  public void add(final long value) {
    if (size == values.length) {
      values = Arrays.copyOf(values, size * 2);
    }
    values[size++] = value;
  }

  /**
   * Returns the value at the specified index in the LongArray.
   *
   * @param index the index of the value to retrieve.
   * @return The value at the specified index in the LongArray.
   * @throws IndexOutOfBoundsException if the index is out of range (index < 0 || index >= size()).
   */
  public long get(final int index) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException(String.format(ERROR_INVALID_INDEX, index, size));
    }
    return values[index];
  }

  /**
   * Returns the current size of the LongArray.
   *
   * @return The current size of the LongArray.
   */
  public int size() {
    return size;
  }

  /**
   * Returns a newly allocated long array containing the current values of the LongArray.
   *
   * @return A newly allocated long array containing the current values of the LongArray.
   */
  public long[] toArray() {
    return Arrays.copyOf(values, size);
  }

}

